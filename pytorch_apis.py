import torch as th
import gp_apis

class gspmmv_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input1, dim_0, dim_1, reverse, norm, device0):
        res = gp_apis.gp_gspmmv(graph, input1, dim_0, dim_1, reverse, norm, device0)
        ctx.save_for_backward(input1)
        ctx.graph = graph
        ctx.dim_0 = dim_0
        ctx.dim_1 = dim_1
        ctx.reverse = reverse
        ctx.norm = norm
        ctx.device0 = device0
        return res

    @staticmethod
    def backward(ctx, grad_output):
        input1, = ctx.saved_tensors
        # Compute the gradient with respect to input1
        # Since our forward operation is essentially a matrix multiplication with the adjacency matrix,
        # the gradient computation involves the transpose of the adjacency matrix.

        # Call the gspmmv function with reverse=True to compute the gradient
        grad_input = gp_apis.gp_gspmmv(ctx.graph, grad_output, ctx.dim_0, ctx.dim_1, not ctx.reverse, ctx.norm, ctx.device0)
        return None, grad_input, None, None, None, None, None


def gspmmv(graph, input1, dim_0, dim_1, reverse, norm, device0):
    return gspmmv_impl.apply(graph, input1, dim_0, dim_1, reverse, norm, device0)
