import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from dataclasses import dataclass

mi.set_variant("llvm_ad_rgb")


class TestBSDF(mi.BSDF):
    def __init__(self, props):
        super().__init__(props)

        self.principled: mi.BSDF = mi.load_dict(
            {
                "type": "principled",
                "base_color": {
                    "type": "rgb",
                    "value": [1.0, 1.0, 1.0],
                },
            }
        )  # type: ignore

    def sample(self, ctx, si, sample1, sample2, active):
        return self.principled.sample(ctx, si, sample1, sample2, active)

    def eval(self, ctx, si, wo, active):
        return 0.0

    def pdf(self, ctx, si, wo, active):
        return 0.0

    def eval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0

    def traverse(self, callback):
        pass

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    def to_string(self):
        pass


mi.register_bsdf("testbsdf", lambda props: TestBSDF(props))

scene = mi.cornell_box()
scene["sensor"]["film"]["width"] = 1024
scene["sensor"]["film"]["height"] = 1024
scene["sphere"] = {
    "type": "sphere",
    "to_world": mi.ScalarTransform4f.translate([0.335, -0.7, 0.38]).scale(0.3),
    "bsdf": {
        "type": "testbsdf",
    },
}
del scene["small-box"]
scene = mi.load_dict(scene)

with dr.suspend_grad():
    img = mi.render(scene)

    # denoiser = mi.OptixDenoiser(img.shape[:2])
    # img = denoiser(img)


plt.imshow(mi.util.convert_to_bitmap(img))
plt.show()
