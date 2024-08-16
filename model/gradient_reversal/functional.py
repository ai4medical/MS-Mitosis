from torch.autograd import Function
# from torch.cuda.amp import custom_bwd,custom_fwd

# class GradientReversal(Function):
#     @staticmethod
#     # @custom_fwd
#     def forward(ctx, x, alpha):
#         ctx.save_for_backward(x, alpha)
#         return x
    
#     @staticmethod
#     # @custom_bwd
#     def backward(ctx, grad_output):
#         grad_input = None
#         _, alpha = ctx.saved_tensors
#         if ctx.needs_input_grad[0]:
#             grad_input = - alpha*grad_output
#         return grad_input, None
# revgrad = GradientReversal.apply

# from torch.autograd import Function


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


revgrad = RevGrad.apply