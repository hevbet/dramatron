"""Font specimen for the Pink Pig 2.0 x Y2K livery. Run: python3 make_specimen.py"""
import os, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
F='/home/user/dramatron/livery/fonts/'
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'specimen.png')

def font(name,size,var=None,axes=None):
    ft=ImageFont.truetype(F+name,size)
    if var:  ft.set_variation_by_name(var)
    if axes: ft.set_variation_by_axes(axes)
    return ft

def mask(text,ft,pad=12):
    l,t,r,b=ft.getbbox(text)
    m=Image.new('L',(r-l+2*pad,b-t+2*pad),0)
    ImageDraw.Draw(m).text((pad-l,pad-t),text,font=ft,fill=255)
    return m

def grad_chrome(w,h):
    y=np.linspace(0,1,h)[:,None]
    st=[(0,245),(0.42,255),(0.5,120),(0.56,70),(0.75,190),(1,250)]
    v=np.interp(y,[s[0] for s in st],[s[1] for s in st])*np.ones((1,w))
    rgb=np.stack([v*0.95+10,v*0.92+8,v*1.0],-1)
    return np.clip(rgb,0,255).astype(np.uint8)

def grad_holo(w,h):
    x=np.linspace(0,1,w)[None,:]; y=np.linspace(0,1,h)[:,None]
    t=(x*1.6+y*0.6)*2*np.pi
    r=200+55*np.sin(t); g=200+55*np.sin(t+2.1); b=215+40*np.sin(t+4.2)
    return np.clip(np.stack([r+0*y,g+0*y,b+0*y],-1),0,255).astype(np.uint8)

def styled(text,ft,style='flat',fill=(25,10,30),outline=(25,10,30),ow=4,shadow=True):
    m=mask(text,ft,pad=ow+14); w,h=m.size
    if style=='chrome': fillimg=Image.fromarray(grad_chrome(w,h))
    elif style=='holo': fillimg=Image.fromarray(grad_holo(w,h))
    else: fillimg=Image.new('RGB',(w,h),fill)
    out=Image.new('RGBA',(w,h),(0,0,0,0))
    if shadow:
        sh=m.filter(ImageFilter.MaxFilter(2*ow+1)) if style!='flat' else m
        sh=sh.filter(ImageFilter.GaussianBlur(5)).point(lambda v:v*0.45)
        out.paste((90,0,40,255),(5,6),sh)
    if style!='flat':
        out.paste(outline+(255,),(0,0),m.filter(ImageFilter.MaxFilter(2*ow+1)))
    out.paste(fillimg,(0,0),m)
    if style in('chrome','holo'):  # sparkle
        d=ImageDraw.Draw(out); cx,cy=w-30,22
        for a,(dx,dy) in enumerate([(14,0),(0,14)]):
            d.polygon([(cx-dx,cy-dy),(cx+3,cy),(cx+dx,cy+dy),(cx-3,cy)] if a==0 else [(cx,cy-dy),(cx+3,cy),(cx,cy+dy),(cx-3,cy)],fill=(255,255,255,255))
        d.polygon([(cx-16,cy),(cx,cy-3),(cx+16,cy),(cx,cy+3)],fill=(255,255,255,255))
    return out

def plate(text,ft,size=(200,150)):
    p=Image.new('RGBA',size,(0,0,0,0)); d=ImageDraw.Draw(p)
    d.rounded_rectangle([0,0,size[0]-1,size[1]-1],22,fill=(255,255,255,255),outline=(25,10,30,255),width=5)
    m=mask(text,ft,pad=0); s=min((size[0]-44)/m.size[0],(size[1]-36)/m.size[1])
    m=m.resize((max(1,int(m.size[0]*s)),max(1,int(m.size[1]*s))),Image.LANCZOS)
    p.paste((25,10,30,255),((size[0]-m.size[0])//2,(size[1]-m.size[1])//2),m)
    return p

W=2600
PINK=(255,122,178); PINK2=(255,170,206); INK=(25,10,30)
rows=[]  # (role, file, varname, axes, size, sample, style, desc)
R=rows.append
R(('NUMBER','Unbounded[wght].ttf','Black',None,'Unbounded Black','wide geometric, very legible at distance'))
R(('NUMBER','RussoOne-Regular.ttf',None,None,'Russo One','classic Russian motorsport digits'))
R(('NUMBER','DelaGothicOne-Regular.ttf',None,None,'Dela Gothic One','fat, poster-like, great in outline'))
R(('NUMBER','Tektur[wdth,wght].ttf',None,[75,900],'Tektur Black (wdth 75)','squared techno, condensed'))
R(('NAME','Exo2-Italic[wght].ttf','Black Italic',None,'Exo 2 Black Italic','clean racing italic (BR03 continuity)'))
R(('NAME','SofiaSansCondensed-Italic[wght].ttf','Black Italic',None,'Sofia Sans Condensed Black Italic','narrow italic: long surnames on roof rail / screen'))
R(('NAME','Unbounded[wght].ttf','Bold',None,'Unbounded Bold','extended, modern'))
R(('SPONSOR','MontserratAlternates-BlackItalic.ttf',None,None,'Montserrat Alternates Black Italic','friendly Y2K-ish geometry, strong italic'))
R(('SPONSOR','Exo2-Italic[wght].ttf','ExtraBold Italic',None,'Exo 2 ExtraBold Italic','neutral partner text, URLs'))
R(('SPONSOR','Tektur[wdth,wght].ttf',None,[100,700],'Tektur Bold','tech look for DriveOil / KARTING64'))
R(('CUT LABEL','YesevaOne-Regular.ttf',None,None,'Yeseva One','vintage butcher-shop fat face'))
R(('CUT LABEL','Podkova[wght].ttf','ExtraBold',None,'Podkova ExtraBold','old shop-sign slab serif'))
R(('CUT LABEL','KellySlab-Regular.ttf',None,None,'Kelly Slab','condensed retro slab, stencil feel'))
R(('Y2K HEADLINE','RubikBubbles-Regular.ttf',None,None,'Rubik Bubbles','bubbly 2000s'))
R(('Y2K HEADLINE','RubikMonoOne-Regular.ttf',None,None,'Rubik Mono One','wide chunky Y2K caps'))
R(('Y2K HEADLINE','StalinistOne-Regular.ttf',None,None,'Stalinist One','extended chrome-logo vibe'))
R(('Y2K HEADLINE','Comfortaa[wght].ttf','Bold',None,'Comfortaa Bold','soft rounded 2000s web'))

SAMPLE={
 'NUMBER':('00','chrome'),
 'NAME':('С. ПОЗДНЯКОВ  ·  М. КОНОПЕЛЬКО','flat'),
 'SPONSOR':('АРКА  ·  СИМКАРТ  ·  DriveOil  ·  KARTING64.RU','flat'),
 'CUT LABEL':('ОКОРОК · ГРУДИНКА · ВЫРЕЗКА · РУЛЬКА','flat'),
 'Y2K HEADLINE':('РОЗОВЫЙ ПОРОСЁНОК 2.0','holo'),
}
ROWH=230
H=200+len(rows)*ROWH+60*5
img=Image.new('RGB',(W,H),PINK); d=ImageDraw.Draw(img)
# butcher dashed lines background
for yy in range(0,H,ROWH):
    pass
lab=font('Exo2-Italic[wght].ttf',30,'Bold Italic'); small=font('Exo2-Italic[wght].ttf',24,'Medium Italic')
title=styled('PINK PIG 2.0 · ШРИФТЫ',font('Unbounded[wght].ttf',96,'Black'),'chrome',outline=INK,ow=5)
img.paste(title,(60,40),title)
y=200; cur=None
for role,fn,var,axes,disp,desc in rows:
    if role!=cur:
        cur=role
        d.rectangle([0,y,W,y+54],fill=INK)
        d.text((60,y+8),'РОЛЬ: '+role,font=lab,fill=(255,255,255))
        y+=60
    d.rectangle([30,y+8,W-30,y+ROWH-8],fill=PINK2)
    # dashed butcher line along bottom
    for x in range(40,W-40,40): d.line([(x,y+ROWH-20),(x+22,y+ROWH-20)],fill=INK,width=4)
    d.text((50,y+20),disp,font=lab,fill=INK); d.text((50,y+58),desc,font=small,fill=(110,20,70))
    d.text((50,y+92),fn,font=small,fill=(110,20,70))
    txt,style=SAMPLE[role]
    x0=620
    if role=='NUMBER':
        ft=font(fn,150,var,axes)
        p=plate('00',font(fn,200,var,axes)); img.paste(p,(x0,y+30),p)
        s=styled('00',ft,'chrome',outline=INK,ow=6); img.paste(s,(x0+240,y+(ROWH-s.size[1])//2),s)
        s=styled('64',ft,'holo',outline=INK,ow=6); img.paste(s,(x0+560,y+(ROWH-s.size[1])//2),s)
        ft2=font(fn,64,var,axes); d.text((x0+900,y+40),'ОКОРОК ГРУДИНКА',font=ft2,fill=INK)
        d.text((x0+900,y+120),'Поздняков 0123456789',font=ft2,fill=(255,255,255),stroke_width=3,stroke_fill=INK)
    else:
        size=90 if role!='Y2K HEADLINE' else 92
        ft=font(fn,size,var,axes)
        while ft.getlength(txt)>W-x0-120: size-=4; ft=font(fn,size,var,axes)
        if style=='flat':
            col=INK if role!='SPONSOR' else (255,255,255)
            if role=='SPONSOR':
                d.text((x0,y+20),txt,font=ft,fill=col,stroke_width=4,stroke_fill=INK)
            elif role=='NAME':
                d.text((x0,y+20),txt,font=ft,fill=INK)
                # flag
            else:
                d.text((x0,y+20),txt,font=ft,fill=INK)
        else:
            s=styled(txt,ft,style,outline=INK,ow=5); img.paste(s,(x0,y+4),s)
        sz2=46; ft2=font(fn,sz2,var,axes)
        while ft2.getlength('ОКОРОК ГРУДИНКА Поздняков 00 · simkart.vercel.app')>W-x0-80: sz2-=2; ft2=font(fn,sz2,var,axes)
        d.text((x0,y+150),'ОКОРОК ГРУДИНКА Поздняков 00 · simkart.vercel.app',font=ft2,fill=(70,10,45))
    y+=ROWH
img.save(OUT); print(OUT,img.size)
