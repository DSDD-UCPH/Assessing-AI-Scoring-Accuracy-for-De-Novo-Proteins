hide everything
bg_color white

show cartoon, target
color forest, target
show surface, target
color gray80, target
set transparency, 0.35, target

select pocket_1, target and chain B and resi 8+9+10+11+12+13+14+15+16+17+18+19+20+21+22+23+24+25+26+27+28+29+30+31
show surface, pocket_1
color red, pocket_1
set transparency, 0.0, pocket_1

select pocket_8, target and chain B and resi 344+345+346+347+348+349+350+351+352+353+354+355+356+357+358+359+360+361+362
show surface, pocket_8
color orange, pocket_8
set transparency, 0.0, pocket_8

select pocket_4, target and chain B and resi 87+88+89+90+91+92+93+96+97+98+99+100+101+102+103+104+105+106+107+108
show surface, pocket_4
color yellow, pocket_4
set transparency, 0.0, pocket_4

select pocket_10, target and chain B and resi 406+407+408+409+410+411+412+413+414+415+416+417+418+419+420
show surface, pocket_10
color cyan, pocket_10
set transparency, 0.0, pocket_10

select pocket_7, target and chain B and resi 315+316+317+318+320+321+322+323+324+325+326+327+328+330+331
show surface, pocket_7
color blue, pocket_7
set transparency, 0.0, pocket_7

select pocket_11, target and chain B and resi 436+437+438+439+440+441+442+443
show surface, pocket_11
color violet, pocket_11
set transparency, 0.0, pocket_11

select pocket_9, target and chain B and resi 380+381+382+383+384+385+386+388
show surface, pocket_9
color magenta, pocket_9
set transparency, 0.0, pocket_9

select pocket_3, target and chain B and resi 64+66+67+68+69+70+71+72
show surface, pocket_3
color salmon, pocket_3
set transparency, 0.0, pocket_3

set cartoon_transparency, 0.15
set surface_quality, 1
set ray_trace_mode, 1
deselect
zoom
