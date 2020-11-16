% Copyright (c) 2018, Plamen P. Angelov and Xiaowei Gu
% All rights reserved. Please read the "license.txt" for license terms.
% This code is the improved version of Autonomous Anomaly Detection Algorithm, which was originally described in:
%==========================================================================================================
%  X. Gu, P. Angelov, �Autonomous anomaly detection�, 
%  in IEEE International Conference on Evolving and Adaptive Intelligent Systems (EAIS), 2017, pp. 1-8.
%==========================================================================================================%% and was modified later in:
%==========================================================================================================
% X. Gu, "Self-organising Transparent Learning System," Phd Thesis, Lancaster University, 2018
%==========================================================================================================
% Please cite the paper and the thesis above if this code helps.
% For any queries about the code, please contact Prof. Plamen P. Angelov and Dr. Xiaowei Gu 
% {p.angelov,x.gu3}@lancaster.ac.uk
% Programmed by Xiaowei Gu
function [Output]= new_test(Input)
% Input
%    Input.Data           - The observed data matrix; each row is a data sample.
% Output
%    Output.IDX	       - The indices of the identified anomalies
%    Output.SystemParams  - The identified anomalies
data=Input;
Lorigin=size(data,1);
Aver=mean(data,1);
X=mean(sum(data.^2,2));
dist1=pdist(data,'euclidean');
Averdist=mean(dist1(find(dist1<=mean(dist1(find(dist1<=mean(dist1)))))));
[UD,J,K]=unique(data,'rows');
F = histc(K,1:numel(J));
[L,W]=size(UD);
GlobalDensity=F./(ones(L,1)+sum((UD-repmat(Aver,L,1)).^2,2)./((X-sum(Aver.^2))));
GlobalDensity=GlobalDensity(K,:);
dist=pdist2(UD,data);
LocalDensity=zeros(L,1);
LPotenAbnorm=round(Lorigin/18);
for i=1:1:L
    s0=find(dist(i,:)<Averdist);
    if length(s0)>1
        data0=data(s0,:);
        Ave0=mean(data0,1);
        DELTA=mean(sum(data0.^2,2))-sum(Ave0.^2);
        LocalDensity(i)=F(i)/(1+sum((UD(i,:)-Ave0).^2)/DELTA)*(length(s0)-1)/(L);
    else
        LocalDensity(i)=0;
    end
end
LocalDensity=LocalDensity(K,:);
[~,IDX1] = sort(LocalDensity,'ascend');
[~,IDX2] = sort(GlobalDensity,'ascend');
IDPA=unique([IDX1(1:1:LPotenAbnorm);IDX2(1:1:LPotenAbnorm)]);
dataPA=data(IDPA,:);
[~,~,IDX,Mnumber,~]=FormingDataCloud(dataPA);
if isempty(Mnumber(Mnumber~=1))~=1
AMN=mean(Mnumber(Mnumber~=1));
else
    AMN=2;
end
seq=find(Mnumber<=AMN);
AbnoID=[];
for i=1:1:length(seq)
    seq0=find(IDX==seq(i));
    AbnoID=[AbnoID;seq0];
end
AbnoIDX=sort(IDPA(AbnoID),'ascend');
AbnoData=data(AbnoIDX,:);
Output.IDX=AbnoIDX;
Output.Anomaly=AbnoData;
end
function [NoC,center,IDX,Mnumber,LocalX]=FormingDataCloud(data)
%%
[L,W]=size(data);
%%
[UD,J,K]=unique(data,'rows');
F = histc(K,1:numel(J));
LU=length(UD(:,1));
%%
dist=pdist(UD,'euclidean');
dist=squareform(dist).^2;
unidata_pi=sum(dist.*repmat(F',LU,1),2);
unidata_density=unidata_pi'*F./(unidata_pi.*2*L);
unidata_glodensity=unidata_density.*F;
[~,pos]=max(unidata_glodensity);
seq=1:1:LU;
seq=seq(seq~=pos);
Rank=zeros(LU,1);
Rank(1,:)=pos;
for i=2:1:LU
    [~,pos0]=min(dist(pos,seq));
    pos=seq(pos0);
    Rank(i,:)=pos;
    seq=seq(seq~=pos);
end
UD1=UD(Rank,:);
UGDen=unidata_glodensity(Rank);
F1=F(Rank);
Gradient=zeros(2,LU-2);
Gradient(1,:)=UGDen(1:1:LU-2)-UGDen(2:1:LU-1);
Gradient(2,:)=UGDen(2:1:LU-1)-UGDen(3:1:LU);
seq2=2:1:LU-1;
seq1=find(Gradient(1,:)<0&Gradient(2,:)>0);
if Gradient(2,LU-2)<0
    seq3=[1,seq2(seq1),LU];
else
    seq3=[1,seq2(seq1)];
end
%%
LU2=length(seq3);
UD2=UD1(seq3,:);
dist1=pdist2(UD2,data);
[~,seq4]=min(dist1,[],1);
centre=zeros(LU2,W);
Mnumber=zeros(LU2,1);
for i=1:1:LU2
    seq5=find(seq4==i);
    Mnumber(i)=length(seq5);
    centre(i,:)=mean(data(seq5,:));
end
seq0=find(Mnumber==1);
M0=length(seq0);
LU2=LU2-M0;
C0=centre(seq0,:);
seq0=find(Mnumber>1);
centre=centre(seq0,:);
Mnumber=Mnumber(seq0);
dist0=pdist2(C0,centre,'euclidean');
[~,id]=min(dist0,[],2);
for i=1:1:M0
    centre(id(i),:)=centre(id(i),:).*Mnumber(id(i))./(Mnumber(id(i))+1)+C0(i,:)./(Mnumber(id(i))+1);
    Mnumber(id(i))=Mnumber(id(i))+1;
end
UD2=centre;
LU3=0;
COUNT=0;
if LU2>2
while LU2~=LU3&&LU2>2
    COUNT=COUNT+1;
    LU3=LU2;
    dist1=pdist2(UD2,data);
    [~,seq4]=min(dist1,[],1);
    centre=zeros(LU2,W);
    Mnumber=zeros(LU2,1);
    sigma=zeros(LU2,1);
    seq7=[];
    for i=1:1:LU2
        seq5=find(seq4==i);
        if length(seq5)>=2
            Mnumber(i)=length(seq5);
            centre(i,:)=mean(data(seq5,:));
            sigma(i)=sqrt(sum(sum(data(seq5,:).^2))/Mnumber(i)-sum(centre(i,:).^2));
            if sigma(i)==0
                seq7=[seq7;i];
            end
        else
            seq7=[seq7;i];
        end
    end
    Mnumber(seq7)=[];
    centre(seq7,:)=[];
    sigma(seq7)=[];
    LU2=LU2-length(seq7);
    dist1=pdist(centre,'euclidean');
    dist3=squareform(dist1);
    cen_pi=sum(dist3.^2.*repmat(Mnumber',LU2,1),2);
    cen_gdensity=cen_pi'*Mnumber./(cen_pi.*2*LU2).*Mnumber;
    seq3=[];
    dist1=pdist(data,'euclidean');
    SIGMA=mean(dist1(find(dist1<=mean(dist1(find(dist1<=mean(dist1)))))));
    dista=dist3-ones(LU2)*SIGMA;
    for i=1:1:LU2
        seq1=[1:1:i-1,i+1:1:LU2];
        seq=seq1(find(dista(i,seq1)<0));
        if isempty(seq)~=1
            if cen_gdensity(i)>max(cen_gdensity(seq))
                seq3=[seq3;i];
            end
        else
            seq3=[seq3;i];
        end
        
    end
    LU2=length(seq3);
    UD2=centre(seq3,:);
end
    [NoC,center,IDX,Mnumber,LocalX]=forming_data_cloud1(UD2,data,SIGMA);
else
%     SIGMA=
    [NoC,center,IDX,Mnumber,LocalX]=forming_data_cloud2(UD2,data);
end
end
function [C,center,IDX,Support,LocalX]=forming_data_cloud1(centre,data,threshold)
dist=pdist2(data,centre,'euclidean');
[L,C]=size(dist);
[~,IDX]=min(dist,[],2);
for i=1:1:C
    seq=find(IDX==i);
    seq2=find(min(dist(seq,:),[],2)>threshold);
    if isempty(seq2)~=1
        centre=[centre;data(seq(seq2),:)];
    end
end
dist=pdist2(data,centre,'euclidean');
[L,C]=size(dist);
[~,IDX]=min(dist,[],2);
for i=1:1:C
    seq=find(IDX==i);
    Support(i)=length(seq);
    LocalX(i)=mean(sum((data(seq,:).^2),2));
    center(i,:)=mean(data(seq,:),1);
end
seq=find(Support==0);
centre(seq,:)=[];
dist=pdist2(data,centre,'euclidean');
[L,C]=size(dist);
[~,IDX]=min(dist,[],2);
for i=1:1:C
    seq=find(IDX==i);
    Support(i)=length(seq);
    LocalX(i)=mean(sum((data(seq,:).^2),2));
    center(i,:)=mean(data(seq,:),1);
end
end
function [C,center,IDX,Support,LocalX]=forming_data_cloud2(centre,data)
dist=pdist2(data,centre,'euclidean');
[L,C]=size(dist);
[~,IDX]=min(dist,[],2);
for i=1:1:C
    seq=find(IDX==i);
    Support(i)=length(seq);
    LocalX(i)=mean(sum((data(seq,:).^2),2));
    center(i,:)=mean(data(seq,:),1);
end
seq=find(Support==0);
centre(seq,:)=[];
dist=pdist2(data,centre,'euclidean');
[L,C]=size(dist);
[~,IDX]=min(dist,[],2);
for i=1:1:C
    seq=find(IDX==i);
    Support(i)=length(seq);
    LocalX(i)=mean(sum((data(seq,:).^2),2));
    center(i,:)=mean(data(seq,:),1);
end
end
