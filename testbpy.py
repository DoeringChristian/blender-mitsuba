from typing import Any
import bpy
import matplotlib.pyplot as plt
import mitsuba as mi
import drjit as dr

mi.set_variant("llvm_ad_rgb")

bpy.ops.wm.open_mainfile(filepath="untitled.blend")


class BlendBSDF(mi.BSDF):
    material_id: str

    def __init__(self, props: mi.Properties) -> None:
        super().__init__(props)
        self.material_id = str(props.get("material"))

    def sample(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        sample1: float,
        sample2: mi.Point2f,
        active: bool = True,
    ) -> tuple[mi.BSDFSample3f, mi.Color3f]:
        material = bpy.data.materials[self.material_id]

        nodes = material.node_tree.nodes
        material_output = nodes["Material Output"]
        print(f"{material_output=}")
        return self.sample_node_input(
            material_output, "Surface", ctx, si, sample1, sample2, active
        )

    def sample_node_input(
        self,
        node: bpy.types.Node,
        input: str,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        sample1: float,
        sample2: mi.Point2f,
        active: bool = True,
    ) -> Any:
        input = node.inputs[input]
        print(f'Sampling node "{node.name}":')
        links = input.links
        if links is not None and len(links) == 1:
            link = links[0]
            name = link.from_socket.name
            input_node: bpy.type.Node = link.from_node
            value = self.sample_node(input_node, ctx, si, sample1, sample2, active)[
                name
            ]
            print(f"{type(value)=}")
            return value
        else:
            print(f"{input.type=}")
            # print(f"{len(input.default_value)=}")
            value = input.default_value
            match input.type:
                case "RGBA":
                    value = mi.Color3f(value[0], value[1], value[2])
                    print(f"{value=}")
                    return value
                case "VECTOR":
                    match input.name:
                        case "Normal":
                            return si.n
                case "VALUE":
                    value = mi.Float(value)
                    print(f"{value=}")
                    return value
                case _:
                    raise Exception(f"Input of type {input.type} is not supported!")

    def sample_node(
        self,
        node: bpy.types.Node,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        sample1: float,
        sample2: mi.Point2f,
        active: bool = True,
    ) -> Any:
        print(f"{node.type=}")
        match node.type:
            case "BSDF_DIFFUSE":
                reflectance = self.sample_node_input(
                    node, "Color", ctx, si, sample1, sample2, active
                )
                normal = self.sample_node_input(
                    node, "Normal", ctx, si, sample1, sample2, active
                )
                print(f"{type(normal)=}")
                print(f"{type(reflectance)=}")
                cos_theta_i = dr.dot(normal, si.wi)

                bs: mi.BSDFSample3f = dr.zeros(mi.BSDFSample3f)

                active &= cos_theta_i > 0.0

                bs.wo = mi.warp.square_to_cosine_hemisphere(sample2)
                bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
                bs.eta = 1.0
                bs.sampled_type = mi.BSDFFlags.DiffuseReflection
                bs.sampled_component = 0

                return {node.outputs[0].name: (bs, mi.Color3f(reflectance))}
            case "VALUE":
                value = node.outputs[0].default_value
                value = mi.Float(value)
                print(f"{value=}")
                return {node.outputs[0].name: value}
            case "RGB":
                value = node.outputs[0].default_value
                value = mi.Color3f(value[0], value[1], value[2])
                print(f"{value=}")
                return {node.outputs[0].name: value}
            case "CAMERA":
                view_vector = si.wi
                view_distance = si.t
                return {
                    node.outputs[0].name: view_vector,
                    node.outputs[2].name: view_distance,
                }
            case "FRESNEL":
                ior = self.sample_node_input(
                    node, "IOR", ctx, si, sample1, sample2, active
                )
                normal = self.sample_node_input(
                    node, "Normal", ctx, si, sample1, sample2, active
                )
                cos_theta_i = dr.dot(normal, si.wi)

                fac = mi.fresnel(cos_theta_i, ior)
                return {node.outputs[0].name: fac[0]}

    def eval(self, ctx, si, wo, active):
        return 0.0

    def pdf(self, ctx, si, wo, active):
        return 0.0

    def eval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0


mi.register_bsdf("blend_bsdf", lambda props: BlendBSDF(props))

bsdfs = {}

for ob in bpy.data.objects:
    if len(ob.material_slots) > 0:
        material_slot = ob.material_slots[0]
        material = material_slot.material
        bsdfs[material.name_full] = mi.load_dict(
            {"type": "blend_bsdf", "material": material.name_full}
        )

scene = mi.cornell_box()
scene["small-box"]["bsdf"] = bsdfs["Material"]
scene = mi.load_dict(scene)

with dr.suspend_grad():
    img = mi.render(scene, spp=1)
plt.imshow(mi.util.convert_to_bitmap(img))
plt.show()
