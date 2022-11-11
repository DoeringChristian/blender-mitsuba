from typing import Any
import bpy
import matplotlib.pyplot as plt
import mitsuba as mi
import drjit as dr
import numpy as np

mi.set_variant("cuda_ad_rgb")

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
                    value = [
                        mi.Float(value[0]),
                        mi.Float(value[1]),
                        mi.Float(value[2]),
                        mi.Float(value[3]),
                    ]
                    print(f"{value=}")
                    return value
                case "VECTOR":
                    return None
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
        ret = {}
        match node.type:
            case "BSDF_DIFFUSE":
                reflectance = self.sample_node_input(
                    node, "Color", ctx, si, sample1, sample2, active
                )
                normal = self.sample_node_input(
                    node, "Normal", ctx, si, sample1, sample2, active
                )
                if normal is None:
                    normal = si.n
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

                ret = {node.outputs[0].name: (bs, mi.Color3f(reflectance[:3]))}
            case "EMISSION":
                color = self.sample_node_input(
                    node, "Color", ctx, si, sample1, sample2, active
                )
                strength = self.sample_node_input(
                    node, "Strength", ctx, si, sample1, sample2, active
                )

                bs: mi.BSDFSample3f = dr.zeros(mi.BSDFSample3f)

                bs.wo = mi.warp.square_to_uniform_sphere(sample2)
                bs.wo = mi.warp.square_to_uniform_sphere_pdf(bs.wo)
                bs.eta = 1.0
                bs.sampled_type = mi.BSDFFlags.Empty
                bs.sampled_component = 0

                ret = {node.outputs[0].name: (bs, mi.Color3f(color * strength))}
            case "VALUE":
                value = node.outputs[0].default_value
                value = mi.Float(value)
                print(f"{value=}")
                ret = {node.outputs[0].name: value}
            case "RGB":
                value = node.outputs[0].default_value
                value = [
                    mi.Float(value[0]),
                    mi.Float(value[1]),
                    mi.Float(value[2]),
                    mi.Float(0.0),
                ]
                print(f"{value=}")
                ret = {node.outputs[0].name: value}
            case "CAMERA":
                view_vector = si.wi
                view_distance = si.t
                ret = {
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
                if normal is None:
                    normal = si.n
                cos_theta_i = dr.dot(normal, si.wi)

                fac = mi.fresnel(cos_theta_i, ior)
                ret = {node.outputs[0].name: fac[0]}
            case "NEW_GEOMETRY":
                position = si.p
                normal = si.n
                tangent = si.to_world(mi.Vector3f(1.0, 0.0, 0.0))
                true_normal = si.n
                incoming = si.wi
                parametric = si.dp_dv
                backfacing = dr.select(
                    dr.dot(si.to_local(si.n), si.wi) < 0, mi.Float(1.0), mi.Float(0.0)
                )
                ret = {
                    node.outputs[0].name: position,
                    node.outputs[1].name: normal,
                    node.outputs[2].name: tangent,
                    node.outputs[3].name: true_normal,
                    node.outputs[4].name: incoming,
                    node.outputs[5].name: parametric,
                    node.outputs[6].name: backfacing,
                }
            case "TEX_COORD":
                generated = si.uv
                normal = si.n
                uv = mi.Point3f(si.uv.x, si.uv.y, 0)
                reflection = mi.reflect(si.wi, si.n)
                ret = {
                    node.outputs[0].name: generated,
                    node.outputs[1].name: normal,
                    node.outputs[2].name: uv,
                    node.outputs[6].name: reflection,
                }
            case "TEX_IMAGE":
                width, height = node.image.size
                pixels = np.array(node.image.pixels)
                # pixels = pixels.reshape((width, height, 4))
                # bm = mi.Bitmap(np.array(node.image.pixels)[..., :3])
                tensor = mi.TensorXf(pixels, shape=[height, width, 4])

                vector = self.sample_node_input(
                    node, "Vector", ctx, si, sample1, sample2, active
                )
                if vector is None:
                    vector = mi.Point3f(si.uv.x, si.uv.y, 0.0)

                match node.interpolation:
                    case "Linear":
                        tex = mi.Texture2f(tensor, filter_mode=dr.FilterMode.Linear)
                        color = tex.eval(mi.Point2f(vector.x, vector.y), active)
                    case "Closest":
                        tex = mi.Texture2f(tensor, filter_mode=dr.FilterMode.Nearest)
                        print(f"{tex.filter_mode()=}")
                        color = tex.eval_cubic(mi.Point2f(vector.x, vector.y), active)
                    case "Cubic":
                        tex = mi.Texture2f(tensor)
                        color = tex.eval_cubic(mi.Point2f(vector.x, vector.y), active)
                    case _:
                        raise Exception(
                            f'Interpolation "{node.interpolation}" not supported!'
                        )

                ret = {node.outputs[0].name: color}
            case _:
                raise Exception(f'Node of type "{node.type} is not supported!"')
        return ret

    def eval(self, ctx, si, wo, active):
        return 0.0

    def pdf(self, ctx, si, wo, active):
        return 0.0

    def eval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0

    def to_string(self):
        return f'BlendBSDF[\n\tmaterial_id = "{self.material_id}"\n]'


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
scene["large-box"]["bsdf"] = bsdfs["Material"]
# print(f"{scene=}")
scene = mi.load_dict(scene)

with dr.suspend_grad():
    img = mi.render(scene)
plt.imshow(mi.util.convert_to_bitmap(img))
plt.show()
