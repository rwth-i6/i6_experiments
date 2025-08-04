import torch
import torch.nn.functional as F
import math


def grad_score_taylor(loss, model, vector_norm_ord=1):
    update_gradient_score = []
    loss.backward(retain_graph=True)
    component_dist = model.component_dist
    model_d = model.conformer.module_list[0].module_list[0].linear_ff_weight.size()[1]
    for i in range(12):
        ff1_num_chunks = component_dist["ff1"][i]
        conv_num_chunks = component_dist["conv"][i]
        mhsa_num_heads = component_dist["mhsa"][i]
        ff2_num_chunks = component_dist["ff2"][i]
        if ff1_num_chunks > 0:
            linear_ff1_grad = (
                model.conformer.module_list[i].module_list[0].linear_ff_weight.grad
                * model.conformer.module_list[i].module_list[0].linear_ff_weight.data
            )
            linear_ff1_grad = torch.reshape(linear_ff1_grad, (ff1_num_chunks, -1, model_d)).abs().detach()
            for j in range(ff1_num_chunks):
                update_gradient_score.append(
                    torch.linalg.vector_norm(linear_ff1_grad[j], ord=vector_norm_ord) / linear_ff1_grad[j].nelement()
                )

        if conv_num_chunks > 0:
            pointwise_conv1_weights = (
                model.conformer.module_list[i].module_list[1].pointwise_conv1_weights.grad
                * model.conformer.module_list[i].module_list[1].pointwise_conv1_weights.data
            )
            pointwise_conv1_weights = (
                torch.reshape(pointwise_conv1_weights, (conv_num_chunks, -1, model_d)).abs().detach()
            )
            for j in range(conv_num_chunks):
                update_gradient_score.append(
                    torch.linalg.vector_norm(pointwise_conv1_weights[j], ord=vector_norm_ord)
                    / pointwise_conv1_weights[j].nelement()
                )

        if mhsa_num_heads > 0:
            mhsa_linear_out_weight = (
                model.conformer.module_list[i].module_list[2].mhsa.linear_out_weight.grad
                * model.conformer.module_list[i].module_list[2].mhsa.linear_out_weight.data
            )
            mhsa_linear_out_weight = torch.reshape(mhsa_linear_out_weight, (model_d, -1, mhsa_num_heads)).abs().detach()
            for j in range(mhsa_num_heads):
                update_gradient_score.append(
                    torch.linalg.vector_norm(mhsa_linear_out_weight[:, :, j], ord=vector_norm_ord)
                    / mhsa_linear_out_weight[:, :, j].nelement()
                )

        if ff2_num_chunks > 0:
            linear_ff2_grad = (
                model.conformer.module_list[i].module_list[3].linear_ff_weight.grad
                * model.conformer.module_list[i].module_list[3].linear_ff_weight.data
            )
            linear_ff2_grad = torch.reshape(linear_ff2_grad, (ff2_num_chunks, -1, model_d)).abs().detach()
            for j in range(ff2_num_chunks):
                update_gradient_score.append(
                    torch.linalg.vector_norm(linear_ff2_grad[j], ord=vector_norm_ord) / linear_ff2_grad[j].nelement()
                )

    update_gradient_score = torch.tensor(update_gradient_score)
    return update_gradient_score


def grad_score_mag(loss, model, vector_norm_ord=1):
    update_gradient_score = []
    loss.backward(retain_graph=True)
    component_dist = model.component_dist
    model_d = model.conformer.module_list[0].module_list[0].linear_ff_weight.size()[1]
    for i in range(12):
        ff1_num_chunks = component_dist["ff1"][i]
        conv_num_chunks = component_dist["conv"][i]
        mhsa_num_heads = component_dist["mhsa"][i]
        ff2_num_chunks = component_dist["ff2"][i]
        if ff1_num_chunks > 0:
            linear_ff1_grad = model.conformer.module_list[i].module_list[0].linear_ff_weight.grad
            linear_ff1_grad = torch.reshape(linear_ff1_grad, (ff1_num_chunks, -1, model_d)).abs().detach()
            for j in range(ff1_num_chunks):
                update_gradient_score.append(
                    torch.linalg.vector_norm(linear_ff1_grad[j], ord=vector_norm_ord) / linear_ff1_grad[j].nelement()
                )

        if conv_num_chunks > 0:
            pointwise_conv1_weights = model.conformer.module_list[i].module_list[1].pointwise_conv1_weights.grad
            pointwise_conv1_weights = (
                torch.reshape(pointwise_conv1_weights, (conv_num_chunks, -1, model_d)).abs().detach()
            )
            for j in range(conv_num_chunks):
                update_gradient_score.append(
                    torch.linalg.vector_norm(pointwise_conv1_weights[j], ord=vector_norm_ord)
                    / pointwise_conv1_weights[j].nelement()
                )

        if mhsa_num_heads > 0:
            mhsa_linear_out_weight = model.conformer.module_list[i].module_list[2].mhsa.linear_out_weight.grad
            mhsa_linear_out_weight = torch.reshape(mhsa_linear_out_weight, (model_d, -1, mhsa_num_heads)).abs().detach()
            for j in range(mhsa_num_heads):
                update_gradient_score.append(
                    torch.linalg.vector_norm(mhsa_linear_out_weight[:, :, j], ord=vector_norm_ord)
                    / mhsa_linear_out_weight[:, :, j].nelement()
                )

        if ff2_num_chunks > 0:
            linear_ff2_grad = model.conformer.module_list[i].module_list[3].linear_ff_weight.grad
            linear_ff2_grad = torch.reshape(linear_ff2_grad, (ff2_num_chunks, -1, model_d)).abs().detach()
            for j in range(ff2_num_chunks):
                update_gradient_score.append(
                    torch.linalg.vector_norm(linear_ff2_grad[j], ord=vector_norm_ord) / linear_ff2_grad[j].nelement()
                )

    update_gradient_score = torch.tensor(update_gradient_score)
    return update_gradient_score


def weight_magnitude_score(model, vector_norm_ord=1):
    update_gradient_score = []
    component_dist = model.component_dist
    model_d = model.conformer.module_list[0].module_list[0].linear_ff_weight.size()[1]
    for i in range(12):
        ff1_num_chunks = component_dist["ff1"][i]
        conv_num_chunks = component_dist["conv"][i]
        mhsa_num_heads = component_dist["mhsa"][i]
        ff2_num_chunks = component_dist["ff2"][i]
        if ff1_num_chunks > 0:
            linear_ff1_weight = model.conformer.module_list[i].module_list[0].linear_ff_weight.data
            linear_ff1_weight = torch.reshape(linear_ff1_weight, (ff1_num_chunks, -1, model_d)).abs().detach()
            for j in range(ff1_num_chunks):
                update_gradient_score.append(
                    torch.linalg.vector_norm(linear_ff1_weight[j], ord=vector_norm_ord)
                    / linear_ff1_weight[j].nelement()
                )

        if conv_num_chunks > 0:
            pointwise_conv1_weights = model.conformer.module_list[i].module_list[1].pointwise_conv1_weights.data
            pointwise_conv1_weights = (
                torch.reshape(pointwise_conv1_weights, (conv_num_chunks, -1, model_d)).abs().detach()
            )
            for j in range(conv_num_chunks):
                update_gradient_score.append(
                    torch.linalg.vector_norm(pointwise_conv1_weights[j], ord=vector_norm_ord)
                    / pointwise_conv1_weights[j].nelement()
                )

        if mhsa_num_heads > 0:
            mhsa_linear_out_weight = model.conformer.module_list[i].module_list[2].mhsa.linear_out_weight.data
            mhsa_linear_out_weight = torch.reshape(mhsa_linear_out_weight, (model_d, -1, mhsa_num_heads)).abs().detach()
            for j in range(mhsa_num_heads):
                update_gradient_score.append(
                    torch.linalg.vector_norm(mhsa_linear_out_weight[:, :, j], ord=vector_norm_ord)
                    / mhsa_linear_out_weight[:, :, j].nelement()
                )

        if ff2_num_chunks > 0:
            linear_ff2_grad = model.conformer.module_list[i].module_list[3].linear_ff_weight.data
            linear_ff2_grad = torch.reshape(linear_ff2_grad, (ff2_num_chunks, -1, model_d)).abs().detach()
            for j in range(ff2_num_chunks):
                update_gradient_score.append(
                    torch.linalg.vector_norm(linear_ff2_grad[j], ord=vector_norm_ord) / linear_ff2_grad[j].nelement()
                )

    update_gradient_score = torch.tensor(update_gradient_score)
    return update_gradient_score


class Conformer_LRP:
    def __init__(self, model, epsilon=1e-4):
        self.model = model
        self.epsilon = epsilon
        self.update_gradient_score = []

    def relevance_linear(self, *, weight, out, input, R_out):
        s = R_out / out
        c1 = torch.matmul(s, weight)
        R_in = c1 * input
        return R_in

    @staticmethod
    def stabilize(input, epsilon=1e-6, clip=False, norm_scale=False, dim=None):

        sign = (input == 0.0).to(input) + input.sign()
        if norm_scale:
            if dim is None:
                dim = tuple(range(1, input.ndim))
            epsilon = epsilon * ((input**2).mean(dim=dim, keepdim=True) ** 0.5)
        if clip:
            return sign * input.abs().clip(min=epsilon)
        return input + sign * epsilon

    def relevance_autograd(self, *, out, input, R_out):
        out = self.stabilize(out)
        print("out torch.isnan(x)", torch.sum(torch.isnan(out)))
        print("min R_out", torch.min(R_out))
        s = R_out / out
        print("s torch.isnan(x)", torch.sum(torch.isnan(s)))
        c = torch.autograd.grad(out, input, grad_outputs=s)[0]
        print("c torch.isnan(x)", torch.sum(torch.isnan(c)))
        R_in = input * c
        print("R_in torch.isnan(x)", torch.sum(torch.isnan(R_in)))
        R_in = R_in / (R_in.sum(dim=-1, keepdim=True))
        print("R_in torch.isnan(x)", torch.sum(torch.isnan(R_in)))
        # R_in = torch.clamp(R_in, min=-1e16, max=1e16)

        return R_in

    def relevance_1d_conv(self, *, conv_werigt, out, input, R_out):
        out += self.epsilon * torch.sign(out)
        relevance_output = R_out / out
        relevance_input = F.conv_transpose1d(relevance_output.transpose(1, 2), conv_werigt, None)
        relevance_input = relevance_input.transpose(1, 2) * input
        return relevance_input

    def relevance_block(self, *, R, block_idx, input_activation):
        block = self.model.conformer.module_list[block_idx]

        # FFN 1
        ff1_input = block.module_list[0].layer_norm(input_activation)
        ff1_z1 = F.linear(ff1_input, block.module_list[0].linear_ff_weight)
        ff1_a1 = block.module_list[0].activation(ff1_z1)
        ff1_z2 = F.linear(ff1_a1, block.module_list[0].linear_out_weight)

        # Conv
        conv_input = block.module_list[1].layer_norm(ff1_z2)
        conv_z1 = F.linear(conv_input, block.module_list[1].pointwise_conv1_weights)
        conv_z2 = torch.nn.functional.glu(conv_z1, dim=-1)
        num_groups = (
            block.module_list[1].depthwise_conv_weights.size()[0]
            // block.module_list[1].depthwise_conv_weights.size()[1]
        )
        conv_z2_T = conv_z2.transpose(1, 2)  # [B,F,T]
        conv_z3 = F.conv1d(
            conv_z2_T,
            block.module_list[1].depthwise_conv_weights,
            None,
            1,
            (block.module_list[1].kernel_size - 1) // 2,
            1,
            num_groups,
        )
        conv_z3_T = conv_z3.transpose(1, 2)  # [B,F,T]
        conv_z4 = F.layer_norm(
            conv_z3_T,
            (block.module_list[1].norm_weight.size(0),),
            block.module_list[1].norm_weight,
            block.module_list[1].norm_bias,
            block.module_list[1].norm_eps,
        )
        conv_z5 = F.linear(conv_z4, block.module_list[1].pointwise_conv2_weights)

        # MHSA
        mhsa_input = block.module_list[2].layernorm(conv_z5)  # [B,T,F]
        # mhsa_input = conv_z5
        n_batch = input_activation.size(0)
        mhsa_q = F.linear(mhsa_input, block.module_list[2].mhsa.linear_q_weight)
        mhsa_k = F.linear(mhsa_input, block.module_list[2].mhsa.linear_k_weight)
        mhsa_v = F.linear(mhsa_input, block.module_list[2].mhsa.linear_v_weight)
        mhsa_q_T = mhsa_q.transpose(1, 2)  # (batch, head, time1, d_k)
        mhsa_k_T = mhsa_k.transpose(1, 2)  # (batch, head, time2, d_k)
        mhsa_v_T = mhsa_v.transpose(1, 2)  # (batch, head, time2, d_k)
        scores = torch.matmul(mhsa_q_T, mhsa_k_T.transpose(-2, -1)) / math.sqrt(block.module_list[2].mhsa.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn.detach(), mhsa_v_T)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(n_batch, -1, block.module_list[2].mhsa.h * block.module_list[2].mhsa.d_k)
        )  # (batch, time1, d_model)
        mhsa_out = F.linear(attn_out, block.module_list[2].mhsa.linear_out_weight)  # (batch, time1, d_model)

        # FFN 2
        ff2_input = block.module_list[3].layer_norm(mhsa_out)
        # ff2_input = mhsa_out
        ff2_z1 = F.linear(ff2_input, block.module_list[3].linear_ff_weight)
        ff2_a1 = block.module_list[3].activation(ff2_z1)
        ff2_z2 = F.linear(ff2_a1, block.module_list[3].linear_out_weight)

        ff2_num_chunks = self.model.component_dist["ff2"][block_idx]

        B, T, _ = ff2_z2.size()

        # FFN 2 LRP with autograd
        ff2_r2 = self.relevance_autograd(out=ff2_z2, input=ff2_a1, R_out=R)
        ff2_score = torch.reshape(ff2_r2, (B, T, -1, ff2_num_chunks))
        ff2_score = torch.sum(ff2_score, dim=(2))
        print("ff2_score", ff2_score)
        print("ff2_score torch.isnan(x)", torch.sum(torch.isnan(ff2_score)))
        ff2_score = torch.mean(ff2_score, dim=(0, 1)) / (1 / ff2_num_chunks)
        self.update_gradient_score = [float(s) for s in ff2_score] + self.update_gradient_score
        print("update_gradient_score", self.update_gradient_score)
        ff2_r1 = self.relevance_autograd(out=ff2_z1, input=ff2_input, R_out=ff2_r2)

        # MHSA LRP with autograd
        mhsa_r3 = self.relevance_autograd(out=mhsa_out, input=attn_out, R_out=ff2_r1)
        print("mhsa_r3 sum", torch.sum(mhsa_r3[0][0]))
        mhsa_r2 = self.relevance_autograd(out=attn_out, input=mhsa_v, R_out=mhsa_r3)
        print("mhsa_r2 sum", torch.sum(mhsa_r2[0][0]))
        mhsa_r1 = self.relevance_autograd(out=mhsa_v, input=mhsa_input, R_out=mhsa_r2)
        print("mhsa_r1 sum", torch.sum(mhsa_r1[0][0]))

        # Conv LRP with autograd
        conv_r4 = self.relevance_autograd(out=conv_z5, input=conv_z4, R_out=mhsa_r1)
        print("conv_r4 sum", torch.sum(conv_r4[0][0]))
        # conv_r3 = self.relevance_autograd(out=conv_z3_T, input=conv_z2, R_out=conv_r4)
        # print("conv_r3 sum", torch.sum(conv_r3[0][0]))
        # print("conv_r3 size", conv_r3.size())
        # conv_r2 = self.relevance_autograd(out=conv_z2, input=conv_z1, R_out=conv_r4)
        # print("conv_r2 sum", torch.sum(conv_r2[0][0]))
        # print("conv_r4", conv_r4[0][0])
        conv_r2 = torch.concat((conv_r4 / 2, conv_r4 / 2), dim=-1)
        # print("conv_r2", conv_r2[0][0])
        print("conv_r2 size", conv_r2.size())
        print("conv_r2 sum", torch.sum(conv_r2[0][0]))
        conv_r1 = self.relevance_autograd(out=conv_z1, input=conv_input, R_out=conv_r2)
        print("conv_r1 sum", torch.sum(conv_r1[0][0]))

        # FFN 1 LRP with autograd
        ff1_r2 = self.relevance_autograd(out=ff1_z2, input=ff1_a1, R_out=conv_r1)
        ff1_r1 = self.relevance_autograd(out=ff1_z1, input=ff1_input, R_out=ff1_r2)
        print("ff1_r1 sum", torch.sum(ff1_r1[0][0]))

        return ff1_r1

    def relevance_ffn(self, R, module, input_activation):
        input_activation = module.layer_norm(input_activation)
        z1 = F.linear(input_activation, module.linear_ff_weight)
        a1 = module.activation(z1)
        z2 = F.linear(a1, module.linear_out_weight)

        # LRP for linear out projection
        # z2 += self.epsilon * torch.sign(z2)  # Add epsilon stabilization term
        # s2 = R / z2  # Relevance distribution
        # c2 = torch.matmul(s2, module.linear_out_weight)  # Contribution from the second layer
        # r2 = z1 * c2

        # LRP with autograd
        z2 += self.epsilon * torch.sign(z2)  # Add epsilon stabilization term
        s2 = R / z2  # Relevance distribution
        c2 = torch.autograd.grad(z2, a1, grad_outputs=s2)[0]
        print("c2", c2)
        r2 = a1 * c2
        print("r2", r2)
        print("r2 sum", torch.sum(r2[0][0]))

        # LRP for linear ff projection
        z1 += self.epsilon * torch.sign(z1)  # Add epsilon stabilization term
        s1 = r2 / z1  # Relevance distribution
        c1 = torch.matmul(s1, module.linear_ff_weight)  # Contribution from the second layer
        r1 = input_activation * c1
        return r1

    def relavance_conv(self, R, module, input_activation):
        input_activation = module.layer_norm(input_activation)
        z1 = F.linear(input_activation, module.pointwise_conv1_weights)
        z2 = torch.nn.functional.glu(z1, dim=-1)
        num_groups = module.depthwise_conv_weights.size()[0] // module.depthwise_conv_weights.size()[1]
        z3 = F.conv1d(
            z2,
            module.depthwise_conv_weights,
            module.depthwise_conv_bias,
            1,
            (module.kernel_size - 1) // 2,
            1,
            num_groups,
        )
        z4 = F.layer_norm(z3, (module.norm_weight.size(0),), module.norm_weight, module.norm_bias, module.norm_eps)
        z5 = F.linear(z4, module.pointwise_conv2_weights, module.pointwise_conv2_bias)

        # LRP with autograd
        r5 = self.relevance_autograd(out=z5, input=z4, R_out=R)
        r4 = self.relevance_autograd(out=z4, input=z3, R_out=r5)
        r3 = self.relevance_autograd(out=z3, input=z2, R_out=r4)
        r2 = self.relevance_autograd(out=z2, input=z1, R_out=r3)
        r1 = self.relevance_autograd(out=z1, input=input_activation, R_out=r2)
        return r1

    def relavance_mhsa(self, R, module, input_activation):
        input_activation = module.layernorm(input_activation)  # [B,T,F]
        n_batch = input_activation.size(0)
        q = F.linear(input_activation, module.mhsa.linear_q_weight, module.mhsa.linear_q_bias).view(
            n_batch, -1, module.mhsa.h, module.mhsa.d_k
        )
        k = F.linear(input_activation, module.mhsa.linear_k_weight, module.mhsa.linear_k_bias).view(
            n_batch, -1, module.mhsa.h, module.mhsa.d_k
        )
        v = F.linear(input_activation, module.mhsa.linear_v_weight, module.mhsa.linear_v_bias).view(
            n_batch, -1, module.mhsa.h, module.mhsa.d_k
        )
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(module.mhsa.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(n_batch, -1, module.mhsa.h * module.mhsa.d_k)
        )  # (batch, time1, d_model)
        out = F.linear(attn_out, module.mhsa.linear_out_weight, module.mhsa.linear_out_bias)  # (batch, time1, d_model)

        # LRP with autograd
        r_out = self.relevance_autograd(out=out, input=attn_out, R_out=R)
        r_v = self.relevance_autograd(out=attn_out, input=v, R_out=r_out)
        r_in = self.relevance_autograd(out=v, input=input_activation, R_out=r_v)
        return r_in

    def get_lrp(self, R, outs):
        final_linear_out = F.linear(outs[-1], self.model.final_linear.weight)
        tmp_R = self.relevance_linear(
            weight=self.model.final_linear.weight, R_out=R, input=outs[-1], out=final_linear_out
        )
        # tmp_R_2 = self.relevance_autograd(out=final_linear_out, input=outs[-1], R_out=R)
        # print("tmp_R", tmp_R[0][0])
        # print("tmp_R_2", tmp_R_2[0][0])
        # print("tmp_R == tmp_R_2", tmp_R[0][0]-tmp_R_2[0][0])

        # tmp_R = self.relevance_final_linear(R, outs[-1])
        for i in range(len(self.model.conformer.module_list) - 1, 9, -1):
            tmp_R = self.relevance_block(R=tmp_R, block_idx=i, input_activation=outs[i - 1])
            tmp_R = tmp_R / (tmp_R.sum(dim=-1, keepdim=True))
            print(tmp_R[0][0])
        # return tmp_R


def obd_score(model, audio_features, audio_features_len, targets, sequence_lengths, targets_len):
    # model.to("cpu")
    # audio_features = audio_features.to("cpu")
    # audio_features_len = audio_features_len.to("cpu")

    def model_loss_fn(audio_features, audio_features_len):
        log_probs, sequence_mask = model(
            audio_features=audio_features,
            audio_features_len=audio_features_len.to("cuda"),
        )
        sequence_lengths = torch.sum(sequence_mask.type(torch.int32), dim=1)

        log_probs = torch.transpose(log_probs, 0, 1)  # [T, B, F]

        loss = torch.nn.functional.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=sequence_lengths,
            target_lengths=targets_len,
            blank=0,
            reduction="sum",
            zero_infinity=True,
        )
        return loss
