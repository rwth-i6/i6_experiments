import torch


def grad_score_taylor(loss, model, vector_norm_ord=1):
    update_gradient_score = []
    loss.backward(retain_graph=True)
    component_dist = model.component_dist
    model_d = model.e_branchformer.module_list[0].ff_1.linear_ff_weight.size()[1]
    for i in range(12):
        ff1_num_chunks = component_dist["ff1"][i]
        cgmlp_num_chunks = component_dist["cgmlp"][i]
        mhsa_num_heads = component_dist["mhsa"][i]
        ff2_num_chunks = component_dist["ff2"][i]
        if ff1_num_chunks > 0:
            linear_ff1_grad = (
                model.e_branchformer.module_list[i].ff_1.linear_ff_weight.grad
                * model.e_branchformer.module_list[i].ff_1.linear_ff_weight.data
            )
            linear_ff1_grad = torch.reshape(linear_ff1_grad, (ff1_num_chunks, -1, model_d)).abs().detach()
            for j in range(ff1_num_chunks):
                update_gradient_score.append(
                    torch.linalg.vector_norm(linear_ff1_grad[j], ord=vector_norm_ord) / linear_ff1_grad[j].nelement()
                )

        if cgmlp_num_chunks > 0:
            linear_ff_weights = (
                model.e_branchformer.module_list[i].cgmlp.linear_ff_weights.grad
                * model.e_branchformer.module_list[i].cgmlp.linear_ff_weights.data
            )
            linear_ff_weights = torch.reshape(linear_ff_weights, (cgmlp_num_chunks, -1, model_d)).abs().detach()
            for j in range(cgmlp_num_chunks):
                update_gradient_score.append(
                    torch.linalg.vector_norm(linear_ff_weights[j], ord=vector_norm_ord)
                    / linear_ff_weights[j].nelement()
                )

        if mhsa_num_heads > 0:
            mhsa_linear_out_weight = (
                model.e_branchformer.module_list[i].mhsa.mhsa.linear_out_weight.grad
                * model.e_branchformer.module_list[i].mhsa.mhsa.linear_out_weight.data
            )
            mhsa_linear_out_weight = torch.reshape(mhsa_linear_out_weight, (model_d, -1, mhsa_num_heads)).abs().detach()
            for j in range(mhsa_num_heads):
                update_gradient_score.append(
                    torch.linalg.vector_norm(mhsa_linear_out_weight[:, :, j], ord=vector_norm_ord)
                    / mhsa_linear_out_weight[:, :, j].nelement()
                )

        if ff2_num_chunks > 0:
            linear_ff2_grad = (
                model.e_branchformer.module_list[i].ff_2.linear_ff_weight.grad
                * model.e_branchformer.module_list[i].ff_2.linear_ff_weight.data
            )
            linear_ff2_grad = torch.reshape(linear_ff2_grad, (ff2_num_chunks, -1, model_d)).abs().detach()
            for j in range(ff2_num_chunks):
                update_gradient_score.append(
                    torch.linalg.vector_norm(linear_ff2_grad[j], ord=vector_norm_ord) / linear_ff2_grad[j].nelement()
                )

    update_gradient_score = torch.tensor(update_gradient_score)
    return update_gradient_score
