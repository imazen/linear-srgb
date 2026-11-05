"""Adapters for typed and table APIs used by generate.py."""
import re

def adapter(item, crate):
    name, args = item[7:].split("(", 1)
    call = crate + "::" + name
    label = crate + "::" + name
    if name.startswith("u16_lut::"):
        return f'{{let y={call}();record("{name}",y.as_slice());}}'
    if name=="alt::accuracy::compare_f32_to_f64_reference<F, G>":
        return '{{let s=ls::alt::accuracy::compare_f32_to_f64_reference(ls::alt::accuracy::naive_srgb_to_linear_f32,ls::alt::accuracy::naive_srgb_to_linear_f64,0.0,1.0,65536);let y=[s.count,s.max_ulp,s.total_ulp as u64,(s.total_ulp>>64) as u64,s.max_ulp_input.to_bits(),s.max_ulp_actual.to_bits(),s.max_ulp_expected.to_bits(),s.avg_ulp().to_bits()];record("alt::accuracy::compare_and_avg",&y);}}'
    if name.startswith("alt::imageflow::SrgbToLinearLut::"):
        method=name.split("::")[-1]
        return f'{{let t=ls::alt::imageflow::SrgbToLinearLut::'+('default' if method=='default' else 'new')+f'();let y:Vec<f32>=u8s.iter().map(|&v|t.lookup(v)).collect();record("{name}",&y);}}'
    if name.startswith("lut::"):
        if "Table<N>::" in name:
            table, method = name.split("::")[1:]
            table = table.replace("<N>", "")
            if method == "new":
                return ""  # exercised by as_slice and lookup, at four sizes
            result = []
            for n in [256, 1024, 4096, 65536]:
                expr = "t.as_slice().to_vec()" if method == "as_slice" else f"(0..{n}).map(|i|t.lookup(i)).collect::<Vec<_>>()"
                result.append(f'{{let t=ls::lut::{table}::<{n}>::new(); let y={expr};record("{name}/{n}",&y);}}')
            return "\n".join(result)
        if name == "lut::lut_interp_linear_float":
            return f'{{let y:Vec<f32>=x.iter().map(|&v|{call}(v,&[0.0,0.25,1.0])).collect();record("{name}",&y);}}'
        if name == "lut::lut_interp_linear_u16":
            return f'{{let y:Vec<u16>=u16s.iter().map(|&v|{call}(v,&[0,16384,65535])).collect();record("{name}",&y);}}'
        if name.startswith("lut::SrgbConverter::"):
            method=name.split("::")[-1]
            if method.startswith("batch_"):
                src,typ=("x","u8") if "linear_to" in method else ("u8s","f32")
                return f'{{let t=ls::lut::SrgbConverter::new();let mut y=vec![0 as {typ};{src}.len()];t.{method}(&{src},&mut y);record("{name}",&y);}}'
            src,typ=("u8s","f32") if method=="srgb_u8_to_linear" else ("x","u8" if method.endswith("u8") else "f32")
            return f'{{let t=ls::lut::SrgbConverter::new();let y:Vec<{typ}>={src}.iter().map(|&v|t.{method}(v)).collect();record("{name}",&y);}}'
    if name.startswith("bytes::") and "u16" in args:
        unpack="rgba1010102_to_rgba16" in name
        source='let s:Vec<u8>=(0..65536u32).flat_map(|v|(v.wrapping_mul(65537)).to_le_bytes()).collect();' if unpack else 'let s:Vec<u16>=(0..262144u32).map(|v|v as u16).collect();'
        typ="u16" if unpack else "u8"
        extra=",256,256,1024,1024" if "strided" in name else ""
        return f'{{{source}let mut y=vec![0 as {typ};262144];{call}(&s,&mut y{extra}).unwrap();record("{label}",&y);}}'
    if name.startswith("deinterleave::"):
        if "chunk" in name:
            count=int(re.search(r"chunk(\d+)",name)[1])
            rgba="rgba" in name
            channels=4 if rgba else 3
            gather="planes_to_" in name
            if gather:
                arg=",".join(['&v']*channels)
                return f'{{let mut y=Vec::new();for chunk in x.chunks_exact({count}){{let v:[f32;{count}]=chunk.try_into().unwrap();y.extend({call}({arg}));}}record("{label}",&y);}}'
            typ="u8" if "rgb24" in name else "u16" if "rgb48" in name else "f32"
            if typ!="f32":
                limit=255 if typ=="u8" else 65535
                code=f'let mut y=Vec::new();for i in 0..={limit}u32 {{let v:[{typ};24]=core::array::from_fn(|k|(i as {typ}).wrapping_add(k as {typ}));let q={call}(&v);y.extend(q.0);y.extend(q.1);y.extend(q.2);}}'
                if "tokenless" in name:
                    return f'#[cfg(target_arch="x86_64")] {{#[archmage::arcane] fn run(t:archmage::X64V3Token)->Vec<f32>{{let _=t;{code}y}} if let Some(t)=archmage::X64V3Token::summon(){{let y=run(t);record("{label}",&y);}}}}'
                return f'{{{code}record("{label}",&y);}}'
            return f'{{let mut y=Vec::new();for chunk in x.chunks_exact({count*channels}){{let q={call}(chunk.try_into().unwrap());y.extend(q.0);y.extend(q.1);y.extend(q.2);'+('y.extend(q.3);' if rgba else '')+f'}}record("{label}",&y);}}'

        gather="planes_f32_to_" in name
        channels=4 if "rgba" in name else 3
        if gather:
            arg=",".join(['&x']*channels)
            suffix=".unwrap()" if "Result<" in args else ""
            return f'{{let mut y=vec![0.0;x.len()*{channels}];{call}({arg},&mut y){suffix};record("{label}",&y);}}'
        typ="u8" if "rgb24" in name else "u16" if "rgb48" in name else "f32"
        src={"u8":"u8s","u16":"u16s","f32":"x"}[typ]
        arg=",".join(f'&mut p{i}' for i in range(channels))
        decl="".join(f'let mut p{i}=vec![0.0;{src}.len()/'+str(channels)+'];' for i in range(channels))
        join="".join(f'y.extend(p{i});' for i in range(channels))
        suffix=".unwrap()" if "Result<" in args else ""
        return f'{{let s=&{src}[..{src}.len()/{channels}*{channels}];{decl}let _={call}(s,{arg}){suffix};let mut y=Vec::new();{join}record("{label}",&y);}}'
    if name.startswith(("typed_rgb::","imgref::")):
        # Convert rustdoc's canonical rgb paths into the public aliases.
        types=re.findall(r"rgb::formats::\w+::(\w+)<(u8|f32)>",args)
        if not types: return None
        def ty(t):return "rgb::"+{"Gray_v08":"Gray","GrayAlpha_v08":"GrayAlpha"}.get(t[0],t[0])+"<"+t[1]+">"
        def width(t):return {"Rgb":3,"Bgr":3,"Rgba":4,"Bgra":4,"Gray_v08":1,"GrayAlpha_v08":2}[t[0]]*(4 if t[1]=="f32" else 1)
        it=ty(types[0]);ib=width(types[0]);ot=ty(types[1]) if len(types)>1 else it;ob=width(types[1]) if len(types)>1 else ib
        setup=f'let raw=bytes_input(65536*{ib},"'+("f32" if types[0][1]=="f32" else "u8")+f'");let mut s:Vec<{it}>=bytemuck::cast_slice(&raw).to_vec();'
        if name.startswith("typed_rgb::"):
            if "&[rgb" in args:
                return f'{{{setup}let mut y:Vec<{ot}>=bytemuck::cast_slice(&vec![0u8;65536*{ob}]).to_vec();{call}(&s,&mut y).unwrap();record("{label}",bytemuck::cast_slice::<{ot},u8>(&y));}}'
            return f'{{{setup}let _={call}(&mut s);record("{label}",bytemuck::cast_slice::<{it},u8>(&s));}}'
        if "ImgVec" in args:
            return f'{{{setup}let y={call}(imgref::ImgVec::new(s,256,256));record("{label}",bytemuck::cast_slice::<{ot},u8>(y.buf()));}}'
        if "ImgRef<'" in args:
            return f'{{{setup}let mut y:Vec<{ot}>=bytemuck::cast_slice(&vec![0u8;65536*{ob}]).to_vec();{call}(imgref::ImgRef::new(&s,256,256),imgref::ImgRefMut::new(&mut y,256,256)).unwrap();record("{label}",bytemuck::cast_slice::<{ot},u8>(&y));}}'
        return f'{{{setup}{call}(imgref::ImgRefMut::new(&mut s,256,256));record("{label}",bytemuck::cast_slice::<{it},u8>(&s));}}'
    if name.startswith(("Convert", "convert")):
        # Generic functions/trait methods are instantiated on RGB(A)→BGR(A).
        if "Image" in name or "imgref" in name:
            if "inplace" in name.lower():
                expr=('<rgb::Rgba<u8> as garb::ConvertImageInplace<rgb::Bgra<u8>>>::convert_image_inplace' if name.startswith('ConvertImageInplace::') else 'garb::convert_imgref_inplace::<rgb::Rgba<u8>,rgb::Bgra<u8>>')+'(imgref::ImgVec::new(s,256,256))'
                return f'{{let raw=bytes_input(65536*4,"u8");let s:Vec<rgb::Rgba<u8>>=bytemuck::cast_slice(&raw).to_vec();let y={expr};record("{label}",bytemuck::cast_slice::<rgb::Bgra<u8>,u8>(y.buf()));}}'
            expr='<rgb::Rgb<u8> as garb::ConvertImage<rgb::Rgba<u8>>>::convert_image' if name.startswith('ConvertImage::') else 'garb::convert_imgref'
            return f'{{let raw=bytes_input(65536*3,"u8");let s:&[rgb::Rgb<u8>]=bytemuck::cast_slice(&raw);let mut y=vec![rgb::Rgba::new(0,0,0,0);65536];{expr}(imgref::ImgRef::new(s,256,256),imgref::ImgRefMut::new(&mut y,256,256)).unwrap();record("{label}",bytemuck::cast_slice::<rgb::Rgba<u8>,u8>(&y));}}' 
        if "inplace" in name.lower():
            expression=('<rgb::Rgb<u8> as garb::ConvertInplace<rgb::Bgr<u8>>>::convert_inplace' if name.startswith('ConvertInplace::') else 'garb::convert_inplace::<rgb::Rgb<u8>,rgb::Bgr<u8>>')+'(&mut s)'
            return f'{{let raw=bytes_input(65536*3,"u8");let mut s:Vec<rgb::Rgb<u8>>=bytemuck::cast_slice(&raw).to_vec();let y={expression};record("{label}",bytemuck::cast_slice::<rgb::Bgr<u8>,u8>(y));}}'
        expr='<rgb::Rgb<u8> as garb::ConvertTo<rgb::Rgba<u8>>>::convert_to' if name.startswith('ConvertTo::') else 'garb::convert'
        return f'{{let raw=bytes_input(65536*3,"u8");let s:&[rgb::Rgb<u8>]=bytemuck::cast_slice(&raw);let mut y=vec![rgb::Rgba::new(0,0,0,0);65536];{expr}(s,&mut y).unwrap();record("{label}",bytemuck::cast_slice::<rgb::Rgba<u8>,u8>(&y));}}'
    return None
