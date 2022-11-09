import torch


class SimpleLossModule(torch.nn.Module):
    def forward(self, *bw_variables):
        # calaulate loss
        loss_sum_cpu = 0
        loss_sum_cuda = None
        for var in bw_variables:
            if var.dtype not in [torch.float16, torch.float32]:
                raise RuntimeError(
                    f"unsupported type for backward variable, dtype: {var.dtype}"
                )

            x = var.flatten()
            # x = torch.where(torch.isnan(x) , torch.full_like(x, 0.), x)
            # x = torch.where(torch.isinf(x) , torch.full_like(x, 0.), x)
            loss_cuda = x.sum()
            if loss_sum_cuda is None:
                loss_sum_cuda = loss_cuda
            else:
                loss_sum_cuda = loss_cuda + loss_sum_cuda

        if loss_sum_cuda is not None:
            loss_sum_cpu += float(loss_sum_cuda.detach().cpu())
            # loss_sum_cpu += float(loss_sum_cuda.float())
            loss = (
                loss_sum_cuda / (2 * loss_sum_cpu + 1e-5) + 1e-4
            )  # so we make sure loss is 0.5, which somehow already make senses.
            return loss
        else:
            # print('No bw_variables detected, return None as loss')
            return None
