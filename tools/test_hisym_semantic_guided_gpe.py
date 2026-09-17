import torch
from model.hisym_semantic_guided_gpe import HiSymSemanticGuidedGPE
torch.manual_seed(2024)
rgb, g25, mask = torch.randn(2,3,256,256), torch.rand(2,1,256,256), torch.zeros(2,1,256,256)
mask[:,:,80:180,90:190] = 1.0
model = HiSymSemanticGuidedGPE(); model.eval()
with torch.no_grad():
    base = model.geometry_fusion(rgb, g25); output = model(rgb,g25,mask)
assert torch.equal(output, base) and model.last_diagnostics['boundary_mean'].item() > 0 and model.last_diagnostics['semantic_delta_abs_mean'].item() > 0
model.train(); model.zero_grad(); model(rgb,g25,mask).square().mean().backward()
assert model.alpha.grad is not None and model.alpha.grad.abs().item() > 0
print('HiSym-SG-GPE functional checks passed')
