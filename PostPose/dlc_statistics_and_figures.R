library(ggplot2)
df <- read.csv("/athena/listonlab/store/dje4001/deeplabcut/processed_video_drop/08_16_2022.csv")

#Custom function
st.err<-function(x){sd(x,na.rm=TRUE)/(sqrt(length(x)))}

#Convert to factors, clean dataframe
df$cage<-as.factor(df$cage)
df$subjectid<-as.factor(df$subjectid)
df$box<-as.factor(df$box)
df$suid<-paste(df$cage,df$subjectid)
df$suid<-as.factor(df$suid)
df$X0<-as.factor(df$X0)

#Calculate distance traveled
dt<-aggreagete(distance~suid,data=df,FUN="sum")
dterr<-

# Aggregate data by Qtip or not
#colnames(df)[9]<-"tmtbool"
# First ten minutes
before_df=df[!(df$time>=2900 & df$time<=3500),]
dist_av_baseline<-aggregate(distance~suid,data=before_df,FUN="mean")
dist_err_baseline<-aggregate(distance~suid,data=before_df,FUN=st.err)
dist_av_baseline$error<-dist_err_baseline$distance
dist_av_baseline$condition<-replicate(length(dist_av_baseline),"baseline")

#During qtip
during_df=df[!(df$time>=3600 & df$time<=4200),]
dist_av_during<-aggregate(distance~suid,data=during_df,FUN="mean")
dist_err_during<-aggregate(distance~suid,data=during_df,FUN=st.err)
dist_av_during$error<-dist_err_during$distance
dist_av_during$condition<-replicate(length(dist_av_during),"QTip")

#Last ten minutes
after_df=df[!(df$time>=4800 & df$time<=5400),]
dist_av_after<-aggregate(distance~suid,data=after_df,FUN="mean")
dist_err_after<-aggregate(distance~suid,data=after_df,FUN=st.err)
dist_av_after$error<-dist_err_after$distance
dist_av_after$condition<-replicate(length(dist_av_after),"postQtip")

dist_av<-rbind(dist_av_baseline,dist_av_during,dist_av_after)
dist_av$condition<-factor(dist_av$condition,levels=c("baseline","QTip","postQtip"))

tmt_mice<-dist_av[(dist_av$suid=="3976646 2" | dist_av$suid=="3976646 3" | dist_av$suid=="3976688 1"),]

tmt_mice_av<-aggregate(distance~condition,data=tmt_mice,FUN="mean")
tmt_mice_err<-aggregate(distance~condition,data=tmt_mice,FUN=st.err)
tmt_mice_av$error<-tmt_mice_err$distance
# Plots of all animal distances
p<-ggplot(data=df,aes(x=time,y=distance))+
  geom_line()+
  facet_grid(X0~suid)
print(p)


# Plot distance from q tip
p<-ggplot(data=dist_av,aes(x=condition,y=distance,ymin=distance-error,ymax=distance+error,group=suid))+
  geom_point()+
  geom_errorbar(width=0)+
  facet_wrap(~suid)
print(p)

p<-ggplot(data=tmt_mice_av,aes(x=condition,y=distance,ymin=distance-error,ymax=distance+error,group=condition))+
  geom_point()+
  geom_errorbar(width=0)
print(p)

#plot of animal distances averaged. 
av_gen_dist<-aggregate(distance~time,data=df,FUN="mean")
er_gen_dist<-aggregate(distance~time,data=df,FUN=st.err)
av_gen_dist$SE<-er_gen_dist$distance
p<-ggplot(data=av_gen_dist,aes(x=time,y=distance))+
  geom_errorbar(aes(ymin=distance-SE, ymax=distance+SE))
print(p)