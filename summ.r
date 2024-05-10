cd /ms/depts/bric/htzhu/ukb_heart/UKB_Heart;module load r;R
sub0=Sys.glob('saved/*_retest')
Feat=c('table_ventricular_volume','table_wall_thickness','table_atrial_volume','table_aortic_area','table_strain_sax','table_strain_lax')
l=length(sub0)
File1=read.csv(paste0('',sub0[1],'/',Feat[1],'.csv'))
File2=read.csv(paste0('',sub0[1],'/',Feat[2],'.csv'))
File3=read.csv(paste0('',sub0[1],'/',Feat[3],'.csv'))
File4=read.csv(paste0('',sub0[1],'/',Feat[4],'.csv'))
File5=read.csv(paste0('',sub0[1],'/',Feat[5],'.csv'))
File6=read.csv(paste0('',sub0[1],'/',Feat[6],'.csv'))
for(i in 2:l)
{
File1=rbind(File1,read.csv(paste0('',sub0[i],'/',Feat[1],'.csv')))
File2=rbind(File2,read.csv(paste0('',sub0[i],'/',Feat[2],'.csv')))
File3=rbind(File3,read.csv(paste0('',sub0[i],'/',Feat[3],'.csv')))
File4=rbind(File4,read.csv(paste0('',sub0[i],'/',Feat[4],'.csv')))
File5=rbind(File5,read.csv(paste0('',sub0[i],'/',Feat[5],'.csv')))
File6=rbind(File6,read.csv(paste0('',sub0[i],'/',Feat[6],'.csv')))
}
write.csv(File1,file=paste0('Final/retest/',Feat[1],'.csv'),quote=F,row.names=F)
write.csv(File2,file=paste0('Final/retest/',Feat[2],'.csv'),quote=F,row.names=F)
write.csv(File3,file=paste0('Final/retest/',Feat[3],'.csv'),quote=F,row.names=F)
write.csv(File4,file=paste0('Final/retest/',Feat[4],'.csv'),quote=F,row.names=F)
write.csv(File5,file=paste0('Final/retest/',Feat[5],'.csv'),quote=F,row.names=F)
write.csv(File6,file=paste0('Final/retest/',Feat[6],'.csv'),quote=F,row.names=F)

QC=NULL
for(i in 2:l)
{
	subtemp=Sys.glob(paste0('',sub0[i],'/out/qc/*.txt'))
	ltemp=length(subtemp)
	if(ltemp>0){
		for(j in 1:ltemp)
		{
			print(c(l-i,j))
			QC=c(QC,read.table(subtemp[j])[[1]])
		}
	}
}
write.table(unique(QC),file=paste0('Final/retest/','qcbad.txt'),row.names=F,col.names=F)
########################################################################################################################################
QC=read.table(paste0('Final/retest/','qcbad.txt'))[[1]]
Feat=c('table_ventricular_volume','table_wall_thickness','table_atrial_volume','table_aortic_area','table_strain_sax','table_strain_lax')
for(i in 1:6)
{
	A=read.csv(paste0('Final/retest/noqc/',Feat[i],'.csv'))
	A=A[-which(A[,1]%in%QC),]
	write.csv(A,paste0('Final/retest/',Feat[i],'_qc.csv'),quote=F,row.names=F)
}



Reprod=NULL
for(i in 1:6)
{
A=read.csv(paste0('Final/retest/',Feat[i],'_qc.csv'))
B=read.csv(paste0('Final/test/',Feat[i],'_qc.csv'));B=B[,-1]
rownames(A)=A[,1];rownames(B)=B[,1];A=A[,-1];B=B[,-1]
ID=intersect(rownames(A),rownames(B))
A=A[ID,];B=B[ID,]
l0=dim(A)[2]
Temp=rep(NA,l0)
names(Temp)=colnames(A)
for(j in 1:l0){
m0=median(A[,j]);m1=mad(A[,j]);m2=median(B[,j]);m3=mad(B[,j]);
ind0=union(which((A[,j]>m0+5*m1)|(A[,j]<m0-5*m1)),which((B[,j]>m2+5*m3)|(B[,j]<m2-5*m3)))
A[ind0,j]=B[ind0,j]=NA
Temp[j]=cor.test(A[,j],B[,j])$est
}
Reprod=c(Reprod, Temp)
}
write.csv(Reprod,file='Reprod.csv',quote=F,row.names=F)

