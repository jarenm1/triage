struct Globals { size_clip: vec4<f32>, transfer: vec4<f32> }
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var rgb: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_2d<f32>;
@group(0) @binding(3) var ids: texture_2d<u32>;
@group(0) @binding(4) var observer: texture_2d<f32>;
@group(0) @binding(5) var atlas: texture_2d<f32>;
struct VertexOut {
 @builtin(position) position: vec4<f32>,
 @location(0) uv: vec2<f32>,
 @location(1) color: vec4<f32>,
 @location(2) @interpolate(flat) params: vec4<f32>,
}
@vertex fn vs_main(@builtin(vertex_index) vertex: u32, @location(0) rect: vec4<f32>, @location(1) color: vec4<f32>, @location(2) params: vec4<f32>) -> VertexOut {
 let corners = array<vec2<f32>,6>(vec2(0.,0.),vec2(1.,0.),vec2(0.,1.),vec2(0.,1.),vec2(1.,0.),vec2(1.,1.));
 let uv = corners[vertex];
 let p = uv * rect.zw;
 let pixel = rect.xy + vec2(cos(params.z)*p.x-sin(params.z)*p.y,sin(params.z)*p.x+cos(params.z)*p.y);
 var out: VertexOut;
 out.position = vec4(pixel/globals.size_clip.xy*vec2(2.,-2.)+vec2(-1.,1.),0.,1.);
 out.uv=uv; out.color=color; out.params=params;
 return out;
}
fn linear(c: vec3<f32>) -> vec3<f32> { return select(pow((c+vec3(.055))/1.055,vec3(2.4)),c/12.92,c<=vec3(.04045)); }
fn srgb(c: vec3<f32>) -> vec3<f32> { return select(1.055*pow(max(c,vec3(0.)),vec3(1./2.4))-vec3(.055),12.92*c,c<=vec3(.0031308)); }
fn palette(id: u32) -> vec3<f32> {
 if id==0u { return vec3(0.); }
 var h=id; h=(h^(h>>16u))*0x7feb352du; h=(h^(h>>15u))*0x846ca68bu; h=h^(h>>16u);
 return vec3<f32>(vec3<u32>(64u)+(vec3<u32>(h,h>>8u,h>>16u)&vec3<u32>(255u))*191u/255u)/255.;
}
@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
 let mode=u32(in.params.x);
 var c=linear(in.color.rgb);
 if mode<=3u {
  let dimensions=select(textureDimensions(rgb),textureDimensions(observer),mode==3u);
  let p=vec2<i32>(min(vec2<u32>(in.uv*vec2<f32>(dimensions)),dimensions-vec2<u32>(1u)));
  if mode==0u { c=textureLoad(rgb,p,0).rgb; }
  if mode==3u { c=textureLoad(observer,p,0).rgb; }
  if mode==1u {
   let id=textureLoad(ids,p,0).r;
   let d=textureLoad(depth,p,0).r;
   let t=clamp((d-globals.size_clip.z)/(globals.size_clip.w-globals.size_clip.z),0.,1.);
   c=linear(vec3(1.-t,1.-abs(2.*t-1.),t));
   if id==0u { c=vec3(0.); }
   else if !(d>=0. || d<0.) || abs(d)>3.402823e38 { c=vec3(1.,0.,1.); }
  }
  if mode==2u { c=linear(palette(textureLoad(ids,p,0).r)); }
 } else if mode==4u {
  let glyph=u32(in.params.y);
  let origin=vec2<u32>(glyph%16u,glyph/16u)*8u;
  let p=origin+min(vec2<u32>(in.uv*8.),vec2<u32>(7u));
  if textureLoad(atlas,vec2<i32>(p),0).r<.5 { c=linear(vec3(16.,19.,25.)/255.); }
 }
 if globals.transfer.x==0. { c=srgb(c); }
 return vec4(c,1.);
}
