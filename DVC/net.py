from subnet import *
import torchac
import cv2
from sklearn.cluster import KMeans


def save_model(model, iter):
    torch.save(model.state_dict(), "./snapshot/iter{}.model".format(iter))


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter') != -1 and f.find('.model') != -1:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])
    else:
        return 0


# The below 3 functions are new functions for modification
def block_matching_motion_estimation(prev_frame, curr_frame, block_size=16, search_range=7):
    height, width = prev_frame.shape[:2]
    motion_vectors = np.zeros((height // block_size, width // block_size, 2), dtype=np.int32)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            best_match = (0, 0)
            min_error = float('inf')

            for m in range(-search_range, search_range + 1):
                for n in range(-search_range, search_range + 1):
                    ref_block_y = i + m
                    ref_block_x = j + n

                    if ref_block_y < 0 or ref_block_y + block_size > height or ref_block_x < 0 or ref_block_x + block_size > width:
                        continue

                    ref_block = prev_frame[ref_block_y:ref_block_y + block_size, ref_block_x:ref_block_x + block_size]
                    curr_block = curr_frame[i:i + block_size, j + j + block_size]

                    error = np.sum((curr_block - ref_block) ** 2)

                    if error < min_error:
                        min_error = error
                        best_match = (m, n)

            motion_vectors[i // block_size, j // block_size] = best_match

    return motion_vectors


def edge_enhancement(image):
    edges = cv2.Canny(image, 100, 200)
    enhanced_image = cv2.addWeighted(image, 0.8, edges, 0.2, 0)
    return enhanced_image


def reduce_color_palette(image, k=16):
    data = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    new_colors = kmeans.cluster_centers_[kmeans.labels_]
    reduced_image = new_colors.reshape(image.shape).astype(np.uint8)
    return reduced_image


class VideoCompressor(nn.Module):
    def __init__(self):
        super(VideoCompressor, self).__init__()
        # self.imageCompressor = ImageCompressor()
        self.opticFlow = ME_Spynet()
        self.mvEncoder = Analysis_mv_net()
        self.Q = None
        self.mvDecoder = Synthesis_mv_net()
        self.warpnet = Warp_net()
        self.resEncoder = Analysis_net()
        self.resDecoder = Synthesis_net()
        self.respriorEncoder = Analysis_prior_net()
        self.respriorDecoder = Synthesis_prior_net()
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.bitEstimator_mv = BitEstimator(out_channel_mv)
        # self.flow_warp = Resample2d()
        # self.bitEstimator_feature = BitEstimator(out_channel_M)
        self.warp_weight = 0
        self.mxrange = 150
        self.calrealbits = False

    def forwardFirstFrame(self, x):
        output, bittrans = self.imageCompressor(x)
        cost = self.bitEstimator(bittrans)
        return output, cost

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe

    def forward(self, input_image, referframe, quant_noise_feature=None, quant_noise_z=None, quant_noise_mv=None):
        # Modified - Convert frames to numpy arrays
        input_image_np = input_image.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC
        referframe_np = referframe.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC

        # Modified - Edge Enhancement
        input_image_np = edge_enhancement(input_image_np)
        referframe_np = edge_enhancement(referframe_np)

        # Modified - Simplified Color Palettes
        input_image_np = reduce_color_palette(input_image_np, k=16)
        referframe_np = reduce_color_palette(referframe_np, k=16)

        # Convert to grayscale if necessary
        if input_image_np.shape[-1] == 3:
            input_image_np = cv2.cvtColor(input_image_np, cv2.COLOR_BGR2GRAY)
            referframe_np = cv2.cvtColor(referframe_np, cv2.COLOR_BGR2GRAY)

        # Perform block matching motion estimation
        estmv = block_matching_motion_estimation(referframe_np, input_image_np)

        # Convert motion vectors to tensor
        estmv_tensor = torch.tensor(estmv, dtype=torch.float32).to(input_image.device)

        # Encode the motion vectors
        mvfeature = self.mvEncoder(estmv_tensor)

        if self.training:
            quant_mv = mvfeature + quant_noise_mv
        else:
            quant_mv = torch.round(mvfeature)

        quant_mv_upsample = self.mvDecoder(quant_mv)

        prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)

        input_residual = input_image - prediction

        feature = self.resEncoder(input_residual)
        batch_size = feature.size()[0]

        z = self.respriorEncoder(feature)

        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)

        recon_sigma = self.respriorDecoder(compressed_z)

        feature_renorm = feature

        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)

        recon_res = self.resDecoder(compressed_feature_renorm)
        recon_image = prediction + recon_res

        clipped_recon_image = recon_image.clamp(0., 1.)

        # Distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        warploss = torch.mean((warpframe - input_image).pow(2))
        interloss = torch.mean((prediction - input_image).pow(2))

        # Bit per pixel
        def feature_probs_based_sigma(feature, sigma):
            def getrealbitsg(x, gaussian):
                cdfs = []
                x = x + self.mxrange
                n, c, h, w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(gaussian.cdf(i - 0.5).view(n, c, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()

                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                return sym_out - self.mxrange, real_bits

            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-5, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))

            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbitsg(feature, gaussian)
                total_bits = real_bits

            return total_bits, probs

        def iclr18_estrate_bits_z(z):
            def getrealbits(x):
                cdfs = []
                x = x + self.mxrange
                n, c, h, w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                return sym_out - self.mxrange, real_bits

            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))

            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbits(z)
                total_bits = real_bits

            return total_bits, prob

        def iclr18_estrate_bits_mv(mv):
            def getrealbits(x):
                cdfs = []
                x = x + self.mxrange
                n, c, h, w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
                return sym_out - self.mxrange, real_bits

            prob = self.bitEstimator_mv(mv + 0.5) - self.bitEstimator_mv(mv - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))

            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbits(mv)
                total_bits = real_bits

            return total_bits, prob

        total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
        total_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
        total_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv)

        im_shape = input_image.size()

        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z + bpp_mv

        return clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp