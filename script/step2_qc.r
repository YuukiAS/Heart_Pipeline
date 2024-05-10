#cd /proj/tengfei/pipeline/UKB_Heart #change to your own path
PATH=/proj/tengfei/pipeline/UKB_Heart/data/:${PATH}
module load r;module load fsl; R
dataPATH='downloaded/nii/tengfei/' #change to your own path
qc='qc'
system(paste0('mkdir ',qc))
sub0=dir(dataPATH)
l=length(sub0)
system(paste0('mkdir ',qc,'/SED/'))
for(i in 1:l)
{
system(paste0('mv ',dataPATH,'/',sub0[i],'/',sub0[i],'/sa_ED.nii.gz qc/SED/',sub0[i],'.nii.gz'))  # SA, ED
system(paste0('mv ',dataPATH,'/',sub0[i],'/',sub0[i],'/seg_sa_ED.nii.gz qc/SED/',sub0[i],'_s.nii.gz'))
}
system(paste0('mkdir ',qc,'/SES/'))
for(i in 1:l)
{
system(paste0('mv ',dataPATH,'/',sub0[i],'/',sub0[i],'/sa_ES.nii.gz qc/SES/',sub0[i],'.nii.gz'))  # SA, ES
system(paste0('mv ',dataPATH,'/',sub0[i],'/',sub0[i],'/seg_sa_ES.nii.gz qc/SES/',sub0[i],'_s.nii.gz'))
}
system(paste0('mkdir ',qc,'/L2ED/'))
for(i in 1:l)
{
system(paste0('mv ',dataPATH,'/',sub0[i],'/',sub0[i],'/la_2ch_ED.nii.gz qc/L2ED/',sub0[i],'.nii.gz'))  # LA2ch, ED
system(paste0('mv ',dataPATH,'/',sub0[i],'/',sub0[i],'/seg_la_2ch_ED.nii.gz qc/L2ED/',sub0[i],'_s.nii.gz'))
}

system(paste0('mkdir ',qc,'/L2ES/'))
for(i in 1:l)
{
system(paste0('mv ',dataPATH,'/',sub0[i],'/',sub0[i],'/la_2ch_ES.nii.gz qc/L2ES/',sub0[i],'.nii.gz'))  # LA2ch, ES
system(paste0('mv ',dataPATH,'/',sub0[i],'/',sub0[i],'/seg_la_2ch_ES.nii.gz qc/L2ES/',sub0[i],'_s.nii.gz'))
}

system(paste0('mkdir ',qc,'/L4ED/'))
for(i in 1:l)
{
system(paste0('mv ',dataPATH,'/',sub0[i],'/',sub0[i],'/la_4ch_ED.nii.gz qc/L4ED/',sub0[i],'.nii.gz'))  # LA4ch, ED
system(paste0('mv ',dataPATH,'/',sub0[i],'/',sub0[i],'/seg_la_4ch_ED.nii.gz qc/L4ED/',sub0[i],'_s.nii.gz'))
}

system(paste0('mkdir ',qc,'/L4ES/'))
for(i in 1:l)
{
system(paste0('mv ',dataPATH,'/',sub0[i],'/',sub0[i],'/la_4ch_ES.nii.gz qc/L4ES/',sub0[i],'.nii.gz'))  # LA4ch, ES
system(paste0('mv ',dataPATH,'/',sub0[i],'/',sub0[i],'/seg_la_4ch_ES.nii.gz qc/L4ES/',sub0[i],'_s.nii.gz'))
}

system(paste0('mkdir ',qc,'/AO/'))
for(i in 1:l)
{
if(i%%100==0)print(l-i)
system(paste0('fslroi ',dataPATH,'/',sub0[i],'/',sub0[i],'/ao.nii.gz qc/AO/',sub0[i],'.nii.gz 0 1'))  # AO
system(paste0('fslroi ',dataPATH,'/',sub0[i],'/',sub0[i],'/seg_ao.nii.gz qc/AO/',sub0[i],'_s.nii.gz 0 1'))
}
########################################################################################################
#QC PNG files generation
########################################################################################################
#1. AO
system(paste0('mkdir ',qc,'/AO_png/'))
for(i in 1:l)
{
if(i%%100==0)print(l-i)
system(paste0('fslmaths qc/AO/',sub0[i],'_s.nii.gz -bin qc/AO/',sub0[i],'_s1.nii.gz'))
system(paste0('slicer.py --edge-overlay qc/AO/',sub0[i],'_s1.nii.gz -e 1 -s a -w 1 qc/AO/',sub0[i],'.nii.gz qc/AO_png/',sub0[i],' --force'))
system(paste0('rm qc/AO/',sub0[i],'_s1.nii.gz'))
}
subpng=dir('qc/AO_png/','.png')
l=length(subpng);
fid0 = file('qc/AO_png/index.html','w')
writeLines('<HTML><TITLE>slicedir</TITLE><BODY BGCOLOR="#aaaaff">',fid0,sep='\n');
Ncol=5
for(i in 1:l){
temp0=paste0('<a href="',sub0[i],'.png','"><img src="',sub0[i],'.png','" WIDTH=200 > ',sub0[i],'.png','</a>')
if(i%%Ncol==0)temp0=paste0(temp0,'<br>')
writeLines(temp0,fid0,sep='\n')
}
writeLines('</BODY></HTML>',fid0,sep='\n');
close(fid0);
#2. L2ES
l=length(sub0) # reset to folder length
system(paste0('mkdir ',qc,'/L2ES_png/'))
for(i in 1:l)
{
if(i%%100==0)print(l-i)
system(paste0('slicer.py --edge-overlay qc/L2ES/',sub0[i],'_s.nii.gz -e 1 -s a -w 1 qc/L2ES/',sub0[i],'.nii.gz qc/L2ES_png/',sub0[i],' --force'))
}

subpng=dir('qc/L2ES_png/','.png')
l=length(subpng);
fid0 = file('qc/L2ES_png/index.html','w')
writeLines('<HTML><TITLE>slicedir</TITLE><BODY BGCOLOR="#aaaaff">',fid0,sep='\n');
Ncol=5
for(i in 1:l){
temp0=paste0('<a href="',sub0[i],'.png','"><img src="',sub0[i],'.png','" WIDTH=200 > ',sub0[i],'.png','</a>')
if(i%%Ncol==0)temp0=paste0(temp0,'<br>')
writeLines(temp0,fid0,sep='\n')
}
writeLines('</BODY></HTML>',fid0,sep='\n');
close(fid0);
#3. L4ES
l=length(sub0)
system(paste0('mkdir ',qc,'/L4ES_png/'))
for(i in 1:l)
{
if(i%%100==0)print(l-i)
system(paste0('slicer.py --edge-overlay qc/L4ES/',sub0[i],'_s.nii.gz -e 1 -s a -w 1 qc/L4ES/',sub0[i],'.nii.gz qc/L4ES_png/',sub0[i],' --force'))
}

subpng=dir('qc/L4ES_png/','.png')
l=length(subpng);
fid0 = file('qc/L4ES_png/index.html','w')
writeLines('<HTML><TITLE>slicedir</TITLE><BODY BGCOLOR="#aaaaff">',fid0,sep='\n');
Ncol=5
for(i in 1:l){
temp0=paste0('<a href="',sub0[i],'.png','"><img src="',sub0[i],'.png','" WIDTH=200 > ',sub0[i],'.png','</a>')
if(i%%Ncol==0)temp0=paste0(temp0,'<br>')
writeLines(temp0,fid0,sep='\n')
}
writeLines('</BODY></HTML>',fid0,sep='\n');
close(fid0);
#4. SES
l=length(sub0)
system(paste0('mkdir ',qc,'/SES_png1/'))
for(i in 1:l)
{
if(i%%100==0)print(l-i)
system(paste0('fslmaths qc/SES/',sub0[i],'_s.nii.gz -uthr 1 -thr 1 -bin qc/SES/',sub0[i],'_s1.nii.gz'))
#system(paste0('slicer.py --edge-overlay qc/SES/',sub0[i],'_s1.nii.gz -e 1 -s a -w 1 qc/SES/',sub0[i],'.nii.gz qc/SES_png1/',sub0[i],' --force'))
system(paste0('slicer.py --overlay qc/SES/',sub0[i],'_s.nii.gz 1 3 -t -e 1 -s a -w 1  qc/SES/',sub0[i],'.nii.gz qc/SES_png1/',sub0[i],' --force'))
system(paste0('rm qc/SES/',sub0[i],'_s1.nii.gz'))
}
subpng=dir('qc/SES_png1/','.png')
l=length(subpng);
fid0 = file('qc/SES_png1/index.html','w')
writeLines('<HTML><TITLE>slicedir</TITLE><BODY BGCOLOR="#aaaaff">',fid0,sep='\n');
Ncol=5
for(i in 1:l){
temp0=paste0('<a href="',sub0[i],'.png','"><img src="',sub0[i],'.png','" WIDTH=200 > ',sub0[i],'.png','</a>')
if(i%%Ncol==0)temp0=paste0(temp0,'<br>')
writeLines(temp0,fid0,sep='\n')
}
writeLines('</BODY></HTML>',fid0,sep='\n');
close(fid0);










