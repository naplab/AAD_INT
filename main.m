
clc;clear;close all

%% figure 1b: INT vs SNR (psychometric curves, from the raw data)
clear Data_Fig1b X Y X2 Y2
X2 = linspace(-12,4,11);

cMap = colororder;
cMap = cMap(3:end,:);

figure;
tiledlayout('flow')
nexttile
noise_cnt = 0;
for noiseType = {'babble','scene'}
    sub_cnt = 0;
    noise_cnt = noise_cnt + 1;
    for sub_each = [28,29,31:42]
        sub_cnt = sub_cnt + 1;
        if sub_each < 30
            load(['..',filesep,'..',filesep,'../../../Psychometric',filesep,'SNRdata',filesep,num2str(sub_each,'%03d'),'psychometric_',noiseType{1},'_PSY.mat'],'allINTs','allSNRs')

        else

            load(['../../../Psychometric',filesep,'SNRdata',filesep,num2str(sub_each,'%03d'),'psychometric_',noiseType{1},'_PSY.mat'],'allINTs','allSNRs')
        end

        allINTs = allINTs * 100;
        X(sub_cnt,:) = allSNRs;
        Y(sub_cnt,:) = allINTs;
        hold on;

        tmp_cnt = 0 ;
        for tmp_ii = X2
            tmp_cnt = tmp_cnt + 1;
            Y2(sub_cnt,tmp_cnt) = getClosestX(allINTs,allSNRs,tmp_ii); % get value for selected SNR
        end
        hold on
        clear tmp*

             plot(X2, Y2(sub_cnt,:),'--','LineWidth',2,'Color',cMap(noise_cnt,:));
        clear allSNRs allINTs


    end
    %average across subject
    hold on
      a(noise_cnt) = errorbar(X2,mean(Y2),std(Y2),'LineWidth',4,'Color',cMap(noise_cnt,:));
    eval(['Data_Fig1b.',noiseType{1},'.SNRs = X2;']);
    eval(['Data_Fig1b.',noiseType{1},'.INTs = Y2;']);
end
legend(a,'Babble','Pedestrian')
ylabel('Speech Intelligibility (%)')
xlabel('SNR (dB)')
grid on
xticks([-12:4:4]);
yticks([0:25:100]);
ylim([0,100]); xlim([-12,4]);
clear a
set(gca,'FontSize',18);

nexttile %average noise
Y3 = cat(3,Data_Fig1b.babble.INTs,Data_Fig1b.scene.INTs);
plot(X2,mean(Y3,3),'LineWidth',2,'Color',0.8*ones(1,3));
hold on
a = errorbar(X2,mean(mean(Y3,3)),std(mean(Y3,3)),'LineWidth',4,'Color','k');
Data_Fig1b.avg.SNRs = X2;
Data_Fig1b.avg.INTs = mean(Y3,3);
legend(a,'Average')
ylabel('Speech Intelligibility (%)')
xlabel('SNR (dB)')
grid on
xticks([-12:4:4]);
yticks([0:25:100]);
ylim([0,100]); xlim([-12,4]);
clear a
set(gca,'FontSize',18);


%% Figure 2A-C: rD vs int and SNR (interpolated figure, best smooth + criteria setting kept)
% data smooth is done for averaged noise, rather than for each noise
% need to manually change rD/rT/rM in this line:  " z = data.aga-data.uga;"
clear
data = load('DataShared_011424.mat');
data = data.t_with_rAU;


clear Xs Ys Zs Zs_smooth Ns Ss
Xs = []; Ys = []; Zs = []; Zs_smooth = []; Ns = []; Ss = [];


%data smooth: assume rD/rT/rM is continuous across intp or snr -> prevent
%the effect from outliers
for sbj_tmp = unique(data.subj)' % for each subject


    idx_tmp = find(data.subj == sbj_tmp);
    z = data.aga-data.uga;
    zName = 'rD';
    xyz = [data.snr(idx_tmp),data.intp(idx_tmp),z(idx_tmp)];
    xyz = sortrows(xyz,1);
    xyz(:,4) = smooth(xyz(:,3),5); % chose 5 samples

    Xs = cat(1,Xs,xyz(:,1));
    Ys = cat(1,Ys,xyz(:,2));
    Zs_smooth = cat(1,Zs_smooth,xyz(:,4));
    Zs = cat(1,Zs_smooth,xyz(:,3));
    Ss = cat(1,Ss,sbj_tmp*ones(size(xyz(:,1))));
    clear xyz idx* z

    Ns = data.noise;

end


clear *_tmp


rD = Zs_smooth;
idx = 1:numel(rD);



[aad_, intp_,snr_,con_] = prepareSurfaceData(rD(idx),Ys(idx),Xs(idx),Ns(idx));
[aad_, intp_,snr_,sbj_] = prepareSurfaceData(rD(idx),Ys(idx),Xs(idx),Ss(idx));


values = [intp_,snr_,aad_,sbj_,con_];
values = sortrows(values);

%..... bin the x axis (SNR) .....
Nx = 15;
xVal = values(:,2);
allx =  min(xVal):(max(xVal)-min(xVal))/Nx:max(xVal);
Nx = numel(allx)-1;
for ii = 1:Nx
    xVal(xVal>allx(ii) & xVal<=allx(ii+1)) = mean([allx(ii),allx(ii+1)]);%ii;
end
ii = 1;
xVal(xVal == allx(1)) = mean([allx(ii),allx(ii+1)]);
% xVal is the binned values(:,2)
allx = mean([allx(1:end-1);allx(2:end)]);


%..... bin the y axis (INT) .....
yVal = values(:,1);
ally = linspace(0,100,16);
Ny = numel(ally)-1;
for jj = 1:Ny
    yVal(yVal>ally(jj) & yVal<=ally(jj+1)) = mean([ally(jj),ally(jj+1)]);%ii;
end
jj = 1;
yVal(yVal == ally(1)) = mean([ally(jj),ally(jj+1)]);
% yVal is the binned values(:,1)
ally = mean([ally(1:end-1);ally(2:end)]);

%..... grouping the rD (zVal) .....
zVal = values(:,3);
[~,id_x,id_y] = findgroups(xVal,yVal);
Nyy = numel(unique(ally)); %number of snr
Nxx = numel(unique(allx)); %number of int
means1 = nan(Nyy,Nxx);
nums1   = nan(Nyy,Nxx);
nums1_avgPerSub_mean = nan(Nyy,Nxx);

ii_cnt = 0;
clear means1_avgPerSub
clear ii jj

for ii = unique(ally)%, y axis variable
    ii_cnt = ii_cnt + 1;
    jj_cnt = 0;
    for jj = unique(allx)%unique(id_intp')
        jj_cnt = jj_cnt + 1;
        ids = intersect(find(yVal==ii),find(xVal==jj));
        rD_vals_current =  [zVal(ids)];
        rD_vals{ii_cnt,jj_cnt} = rD_vals_current;
        sbj_vals{ii_cnt,jj_cnt} = [values(ids,4)];

        intp1(ii_cnt,jj_cnt) = ii;
        snr1(ii_cnt,jj_cnt) = jj;
        intp2{ii_cnt,jj_cnt} = ii*ones(numel(rD_vals_current),1); %distribution
        snr2{ii_cnt,jj_cnt} = jj*ones(numel(rD_vals_current),1); %distribution
        intp_true{ii_cnt,jj_cnt} = [values(ids,1)];
        snr_true{ii_cnt,jj_cnt} = [values(ids,2)];

        nums1(ii_cnt,jj_cnt) = numel(ids);

        means1(ii_cnt,jj_cnt) = mean(rD_vals_current);
        grouping = findgroups(sbj_vals{ii_cnt,jj_cnt});

        if ~isempty(rD_vals{ii_cnt,jj_cnt})
            nums1_avgPerSub_mean(ii_cnt,jj_cnt) = numel(splitapply(@mean,rD_vals{ii_cnt,jj_cnt},grouping));
        end



    end

end
clear ii* jj*

%at least 10 trials from 3 different subjects for each condition
criteria = double(nums1_avgPerSub_mean>=3).*double(nums1>=10);
criteria(criteria==0) = NaN;


% Figure 2A-2C: plot means (best setting)

figure;
acc = means1.*criteria;
acc = smoothdata2(acc,'movmean',{3,3});

imagesc(acc);colormap(jet)
axis xy;title(['mean (',zName,')']);c = colorbar; axis square;
ylabel('Intelligibility (%)'); xlabel('SNR (dB)')
set(gca,'FontSize',22);
cmap = jet;
cmap(1,:) = [1,1,1];
colormap(cmap)
title(c,[zName,' (a.u.)'])
caxis([-0.01,0.03])

if Nyy<=10
    yticks([1:Nyy]);
    yticklabels(round(ally([1:Nyy]),2))
else
    yticks([2:2:Nyy]);
    yticklabels(round(ally([2:2:Nyy]),2))
end


if Nxx<=10
    xticks([1:Nxx]);
    xticklabels(round(allx([1:Nxx]),2))
else
    xticks([2:2:Nxx]);
    xticklabels(round(allx([2:2:Nxx]),2))
end
clim([prctile(acc(:),0)-0.01,prctile(acc(:),95)])

clear bts* cMap*  cmap c c3 f1 f2 id* gaps *_ *1 *2 *avgPerSub* t2 USE* values idx SHOW*
clear Xs Ys Zs* Ns Ss *Val *true *vals Nx* Ny* p3 *_btrp* criteria  ans allx ally
clear select* medians rD stds ste xyz
clear acc zName

%% Figure 2d/e [done]  - rD/rT/rM vs INT/SNR (averaged across subjects first)

clearvars -except data
% ---- initialization ----
% if we want to display all three correlations on one single plot, manually
% do that.

idx = 1:size(data,1);
BST_EACHSUB = 1; %boostrap and smooth rD/rA/rU for each subject
SMOOTH_EACHSUB = 1;

% -------------

Y = data.rD_original;% or data.aga, data.uga
YName = 'rD'; % or rT or rM

sbj = data.subj;
intp = data.intp;
hitP = data.hitP;
snr = data.snr  ;
da = data.da;
con = data.noise;


[aad_, sbj_,int_,hitp_] = prepareSurfaceData(Y(idx),sbj(idx),intp(idx),hitP(idx));
[aad_, sbj_,snr_,hitp_] = prepareSurfaceData(Y(idx),sbj(idx),snr(idx),hitP(idx));
[aad_, sbj_,da_,hitp_] = prepareSurfaceData(Y(idx),sbj(idx),da(idx),hitP(idx));
[aad_, sbj_,hitp_,con_] = prepareSurfaceData(Y(idx),sbj(idx),hitP(idx),con(idx));

clear values*
values = [int_,snr_,sbj_,con_,aad_,da_,nan(size(aad_)),hitp_];
values = sortrows(values);

if SMOOTH_EACHSUB
    value_smooth = [];
    for sbj_cnt = unique(sbj_)'
        values_ = values(values(:,3) == sbj_cnt,:);
        values_ = sortrows(values_);
        values_(:,7) = smooth(values_(:,5),10); %10 samples this time
        value_smooth = cat(1,value_smooth,values_);
        clear values_
    end
    values = sortrows(value_smooth);

    clear  value_smooth sbj_cnt
end
values_unbin = values;
clear *_ idx


XvalName = {'SI (%)','SNR (dB)'}; % the order is fixed
for XvalCnt =  1:numel(XvalName)
    figure;

    if XvalCnt == 1
        % allX = [0,20:10:100];
        allX = linspace(0,100,11);


    else
        N1 = 10;
        allX = [min(values(:,2)):(max(values(:,2))-min(values(:,2)))/N1:max(values(:,2))];
        % allX = [-12:2:4];
        N1 = numel(allX)-1;
    end



    clear meanVal*
    meanVal = nan(numel(unique(values(:,3))),N1); %subject * bins



    ii = 0;
    for sub_each =  unique(values(:,3))' % each subject

        ii = ii + 1;
        clear ids_sub
        ids_sub = values(:,3) == sub_each;

        for jj = 1:numel(allX)-1 % each bin

            clear ids ids_bin ids_babble ids_street

            ids_bin = (values(:,XvalCnt)>allX(jj)) & (values(:,XvalCnt)<=allX(jj+1));
            values(ids_bin,XvalCnt) = mean([allX(jj),allX(jj+1)]);

            ids = ids_sub & ids_bin;


            if BST_EACHSUB % bootstrap for each subject
                if SMOOTH_EACHSUB
                    fea_col = 7; %smoothed correlation
                else
                    fea_col = 5; % raw correlation
                end


                samples = values(ids,fea_col);
                if  ~isempty(samples)
                    for bts_time = 1:100
                        bts_samples(bts_time) = mean(randsample(samples,10,true));
                    end
                    meanVal(ii,jj) = mean(bts_samples);
                end
                clear bts_samples bts_time samples


            else %no boostrap
                if SMOOTH_EACHSUB
                    fea_col = 7;
                else
                    fea_col = 5;
                end
                meanVal(ii,jj)        = mean(values(ids,fea_col),'omitnan');
            end
            clear fea_col


        end
    end
    allX = mean([allX(1:end-1);allX(2:end)]);

    % average across eubjct
    meanVal_avg = mean(meanVal,'omitnan');
    % 95% ci  (2*SE)
    sub_num_all = sum(~isnan(meanVal),1);
    ciVal_avg_all =  2*std(meanVal,[],1,'omitnan')./sqrt(sub_num_all);



    % plot all
    hold on
    x = allX;
    y = meanVal_avg;
    errbar = ciVal_avg_all;
    pop.col = {[0.2,0.2,0.2]};
    % pop.col = {[0.8,0.8,0.8]};
    pop.width = 4;

    h = mseb_no_edge(x,y,errbar,pop,0.2);

    clear x y errbar pop


    % mark the 0 line
    hold on
    yline(0,'--','LineWidth',1)


    % format
    grid on

    ax = gca; ax.FontSize = 18;
    gaps = (allX(end) - allX(1))/(numel(allX)-1)/2;
    xlim([allX(1)-gaps,allX(end)+gaps])
    xticks([round(allX*10000)/10000])
    xlabel(XvalName{XvalCnt});

    ylabel(['Mean(',YName,') +/- 95% CI'])
    title(['Mean (',YName,') across ',XvalName{XvalCnt}],'FontSize',22);



    clear ax gaps
    clear meanval pop  x_* y stes grouping id_2 id_babble_full a
    clear h_*





end



clear SHOW* MEAN idx values *_ *Cnt XvalName sub_each


%% Figure 2F/G: for each SI, show the effect on rD from SNR
% LME for fixed-term analysis
clc;clear
load('./DataShared_Revision_020625.mat');
clearvars -except data*
clearvars -except data_*
clc
formula = 'rD ~ SNR +rINT+ (1|subj)';
disp(formula)

data_used1 = data_fit;
data_used2 = data_full;

% factor to be controlled
xA_Name = 'SI';
xA = data_used1.rINT;
[xA1,xA2] = discretize(xA,[0:20:100]); 
xA_bin = xA2(xA1);
clear xA1 xA2

xB = data_used1.SNR;
xB_Name = 'SNR';

y_Name = 'rD';  
Y = data_used1.rD;

% in the full range of SNR, fit lme
figure('Position',[312 44 443 664])
tiledlayout(8,1,'TileSpacing','tight')
nexttile([4,1])
xA_cnt  = 0 ;
cmap = [jet(numel(unique(xA_bin)));[0,0,0]];
% cmap = jet(15);
for xA_now = [unique(xA_bin),9999]
    xA_cnt = xA_cnt + 1;
    if xA_now == 9999
        % the full SI
        tmp_lme = fitlme(data_used2,formula, 'FitMethod', 'ML');
        x_pred = linspace(min(data_used2.SNR),max(data_used2.SNR),100)';
  
    else
        index1 = find(xA_bin == xA_now);
        tmp_lme = fitlme(data_used1(index1,:),formula, 'FitMethod', 'ML');

        % plot the fixed line +/- se from random lines
        x_pred = linspace(min(xB(index1)),max(xB(index1)),100)';
    end

    hold on
      indx = find(strcmpi(tmp_lme.CoefficientNames,'SNR'));
    Coeffs = tmp_lme.fixedEffects;
    fixedIntercept(xA_cnt)  = Coeffs(1);
    fixedSlope(xA_cnt)  = Coeffs(indx);
    Coeffs_p = tmp_lme.Coefficients.pValue;
    pval(xA_cnt) = Coeffs_p(indx);

 
    [randomIntercepts,~,randomIntercepts_table] = randomEffects(tmp_lme);
    randomPval = randomIntercepts_table.pValue;

      indx = find(strcmpi(tmp_lme.CoefficientNames,'rINT'));
    Coeffs = tmp_lme.fixedEffects;
    fixedSlope_int  = Coeffs(indx);
    if xA_now<999
        y_fixed(:,xA_cnt) =  fixedIntercept(xA_cnt) + fixedSlope(xA_cnt) .* x_pred + fixedSlope_int .*mean(data_used1(index1,:).rINT);
    else
        y_fixed(:,xA_cnt) =  fixedIntercept(xA_cnt) + fixedSlope(xA_cnt) .* x_pred + fixedSlope_int .*mean(data_used2.rINT);
    end

    sub_list = unique(data_used1.subj);
    for sub_cnt = 1:14
        clear indx*
        indx1 = find(strcmpi(randomIntercepts_table.Name,'(Intercept)'));
        indx2 = find(strcmpi(randomIntercepts_table.Level,num2str(sub_list(sub_cnt))));
        indx_intercept = intersect(indx1,indx2);

        clear indx1 indx2
        indx1 = find(strcmpi(randomIntercepts_table.Name,'SNR'));
        indx2 = find(strcmpi(randomIntercepts_table.Level,num2str(sub_list(sub_cnt))));
        indx_slope = intersect(indx1,indx2);

        % get offset
        if and(isempty(indx_slope),~isempty(indx_intercept)) %(1|subj)
            groupOffset(sub_cnt) = randomIntercepts(indx_intercept); % 该组的随机截距
            groupOffset_pval(sub_cnt,xA_cnt) = randomPval(indx_intercept);
        elseif and(~isempty(indx_intercept),~isempty(indx_slope)) % (SNR|sbj)
            groupOffset(sub_cnt) = randomIntercepts(indx_intercept); % 该组的随机截距
            groupOffset_pval(sub_cnt,xA_cnt) = randomPval(indx_intercept); %随机截距的pval
        else
            %may be missing this subject or no random term
             groupOffset(sub_cnt) = 0;
             groupOffset_pval(sub_cnt,xA_cnt) = nan;
        end


        if ~isempty(indx_slope) %(SNR|subj)
            groupSlope(sub_cnt) = randomIntercepts(indx_slope);
             y_random(:,sub_cnt) = (fixedIntercept(xA_cnt) + groupOffset(sub_cnt)) + (fixedSlope(xA_cnt) + groupSlope(sub_cnt)) .* x_pred; % 该组的拟合线

             allSlope(sub_cnt) = (fixedSlope(xA_cnt) + groupSlope(sub_cnt));
             allIntercept(sub_cnt) =  (fixedIntercept(xA_cnt) + groupOffset(sub_cnt));
        else %(1|subj) or no random term
            
            y_random(:,sub_cnt) = (fixedIntercept(xA_cnt) + groupOffset(sub_cnt)) + (fixedSlope(xA_cnt)) .* x_pred; % 该组的拟合线

            allSlope(sub_cnt) = (fixedSlope(xA_cnt));
            allIntercept(sub_cnt) =  (fixedIntercept(xA_cnt) + groupOffset(sub_cnt));
        end

        
       



    end

    meanAllSlope(xA_cnt) = mean(allSlope);
    meanAllIntercept(xA_cnt) = mean(allIntercept);
    if xA_now~=10000
        % % y_fixed(:,xA_cnt) = fixedIntercept(xA_cnt) + fixedSlope(xA_cnt) .* x_pred; % 固定效应拟合线
        % y_fixed(:,xA_cnt) = mean(y_random,2);
        % y_fixed(:,xA_cnt) =  meanAllSlope(xA_cnt).*x_pred + meanAllIntercept(xA_cnt);
        hold on
        clear x y e pop h
        x = x_pred';
        y = y_fixed(:,xA_cnt)';
        e = std(y_random,[],2)./sqrt(14); e = e';
        pop.col = {cmap(xA_cnt,:)};
        % pop.col = {'k'};
        pop.width = 5;
        h = mseb_no_edge(x,y,e,pop,0);

        hold on
        if xA_now <999
            scatter(data_used1(index1,:).SNR,data_used1(index1,:).rD,[],cmap(xA_cnt,:),'filled','MarkerFaceAlpha',0.2);
        else
            scatter(data_used2.SNR,data_used2.rD,[],cmap(xA_cnt,:),'filled','MarkerFaceAlpha',0.2);
        end
        hold on
    else
        hold on
        clear x y e pop h
        x = x_pred';
        y = y_fixed(:,xA_cnt)';
        e = std(y_random,[],2)./sqrt(14); e = e';
        pop.col = {[0,0,0]};
        pop.width = 5;
        pop.stype = ':';
        h = mseb_no_edge(x,y,e,pop,0);
    end
   

end


xlabel('SNR');
ylabel('mean(rD) +/- SE(rD)');

title('LME: fixed effect & subject effect');
hold off;
set(gca,'FontSize',12)
grid on
ylim([-0.6,0.8])


nexttile([2,1])
bar([0:20:100],[meanAllSlope(1:end);fixedSlope(1:end)])
legend({'Mean Slope (fixed + random)','Fixed Effect'})
hold on

labels = num2cell([0:20:100]);
labels{end} = 'full';

xticklabels(labels)
clear all_slope_ labels
ylabel('Slope');
title('Slope of GV across SNR',formula)

nexttile([2,1])
colormap(jet(10))
bar([0:20:100]',[meanAllSlope(1:end)]');
legend({'Mean Slope (fixed + random)','Fixed Effect'})
hold on
bar([0:20:100],meanAllSlope.*double(pval<0.05),'DisplayName','Significant')
labels = num2cell([0:20:100]);
labels{end} = 'full';


xticklabels(labels)
clear all_slope_ labels
ylabel('Slope');
title('Slope of GV across SNR',formula)

%% Figure 3A/B: GV vs SNR
% average across subject (for each subject first, and across subjects)


clear data_used
data_used = data_fit;
y_fea = data_used.EyeMovVel;
y_fea_name = 'GV';
x_fea = data_used.SNR;
x_fea_name = 'SNR';



% ..... process the x_fea (discretize)  .......
if strcmpi(x_fea_name,'SNR')
    allX = [-12:2:4];
elseif or(strcmpi(x_fea_name,'INT'),contains(x_fea_name,'INT'))
    allX = [0,20:10:100];
end
Nx = numel(allX)-1;

x_fea_bin = x_fea;

if strcmpi(y_fea_name,'HR')
    MEAN = false;
else
    MEAN = true;
    SMOOTH = true;
end

% smooth GV for each subject, 5 samples
if SMOOTH
    XYZ = [];
    % xyz = [data_used.SNR,data_used.EyeMovVel,data_used.subj];
    xyz = table2array(data_used);
    for z = unique(xyz(:,4))' %the subject list
        ids = find(xyz(:,4)==z);
        XYZ_new = sortrows(xyz(ids,:),2);
        XYZ_new(:,3) = smooth(XYZ_new(:,3),10);

        XYZ = cat(1,XYZ,XYZ_new); % sort on the SNR (column 2)
        % XYZ(:,3) = smooth(XYZ(:,3),5); % only smooth GV
        clear XYZ_new
    end

    x_fea = XYZ(:,2);
    y_fea = XYZ(:,3);


    data_tmp_GVsmooth = array2table(XYZ,'VariableNames',data_used.Properties.VariableNames);
    clear XYZ xyz ids z

    data_used = data_tmp_GVsmooth;
end


% leave space for eac hsubject
clear meanVal*
meanVal = nan(numel(unique(data_used.subj),Nx)); %subject * bins
meanVal_num = nan(size(meanVal)); %subject * bins


sub_cnt = 0;
for sub_each = unique(data_used.subj)' % each subject
    sub_cnt = sub_cnt + 1;
    clear ids_sub
    ids_sub =data_used.subj == sub_each;

    for x_cnt = 1:numel(allX)-1 % each bin
        clear ids ids_bin
        ids_bin = (x_fea>allX(x_cnt)) & (x_fea<=allX(x_cnt+1));
        x_fea_bin(ids_bin) = mean([allX(x_cnt),allX(x_cnt+1)]);
        ids = ids_sub & ids_bin;

        if MEAN

            meanVal(sub_cnt,x_cnt)        = mean(y_fea(ids),'omitnan');
            meanVal_num(sub_cnt,x_cnt)    = numel(find(ids));

        else
            medianVal(sub_cnt,x_cnt)        = median(y_fea(ids),'omitnan');

        end
    end


end

allX = mean([allX(1:end-1);allX(2:end)]);

if MEAN % SE
    avgVal = mean(meanVal,'omitnan');
    sub_num_all = sum(~isnan(meanVal),1);
    errVal_avg_all =  std(meanVal,[],1,'omitnan')./sqrt(sub_num_all);

    p = nan(1,size(meanVal,2));
    for bin_cnt = 1:size(p,2)
        [~,p(bin_cnt)] = ttest(meanVal(:,bin_cnt),0,'tail','right');
    end
else
    avgVal = mean(medianVal,'omitnan');
    sub_num_all = sum(~isnan(medianVal),1);
    errVal_avg_all =  2*std(medianVal,[],1,'omitnan')./sqrt(sub_num_all);

    p = nan(1,size(medianVal,2));
    for bin_cnt = 1:size(p,2)
        [~,p(bin_cnt)] = ttest(medianVal(:,bin_cnt),0,'tail','right');
    end
end

clear bin_cnt cmap

% plot (average across)
figure('Position',[102 221 625 505]);
x = allX;
y = avgVal;
errbar = errVal_avg_all;
pop.col = {'k'};
pop.width = 4;
h(3) = mseb_no_edge(x,y,errbar,pop,0);
% [c,p] = corr(allX)
clear x y errbar pop

% format
grid on
ax = gca; ax.FontSize = 18;
gaps = (allX(end) - allX(1))/(numel(allX)-1)/2;
xlim([allX(1)-gaps,allX(end)+gaps])
xticks([round(allX*10000)/10000])
xlabel(x_fea_name);
if MEAN
    ylabel(['Mean(',y_fea_name,') +/- SE(',y_fea_name,')'])
    title(['Mean ',y_fea_name,' across ',x_fea_name],'FontSize',22);
else
    ylabel(['Median(',y_fea_name,') +/- 95% CI'])
    title(['Median  ',y_fea_name,' across ',x_fea_name],'FontSize',22)
end

clearvars -except data*

%% Figure 3C/D: rD vs GV in Celing SI

clearvars -except data_*
data_used = data_full;
tmp_lme = fitlme(data_used,'rD~EyeMovVel + (EyeMovVel|subj)', 'FitMethod', 'ML');%best


% get slopes and intercept for each subjects
X = data_used.EyeMovVel;
Y = data_used.rD;
index = and(X>0,X<50);
X = X(index);
Y = Y(index);

X_pred = linspace(min(X), max(X), 100)';
fixedIntercept = fixedEffects(tmp_lme); % 截距
fixedIntercept = fixedIntercept(1);
fixedSlope = fixedEffects(tmp_lme); % 斜率
fixedSlope = fixedSlope(2);

[randomIntercepts,~,randomIntercepts_table] = randomEffects(tmp_lme);
randomPval = randomIntercepts_table.pValue;

figure('Position',[1 287 1440 268]); hold on;
tiledlayout('flow')
nexttile

scatter(X,Y,[],[0,0,0],'filled','MarkerFaceAlpha',0.2)
xlabel('GV')
ylabel('rD')
title('scatter plot on full SI')
set(gca,'FontSize',12)

% fixed effect (overall)
y_fixed = fixedIntercept + fixedSlope * X_pred; % 固定效应拟合线


% random effect for each subject
nexttile
hold on
colors = jet(14);

% yyaxis right
for i = 1:14

    groupSlope(i) = randomIntercepts(i*2);
    groupSlope_p(i) = randomPval(i*2);
    groupOffset(i) = randomIntercepts(i*2-1); % 该组的随机截距
    groupOffset_p(i) = randomPval(i*2-1);
    y_random(:,i) = (fixedIntercept + groupOffset(i)) + (fixedSlope + groupSlope(i)) * X_pred; % 该组的拟合线

    plot(X_pred, y_random(:,i), 'Color', [0.8,0.8,0.8], 'LineWidth', 1.5);
end
hold on
x = X_pred';
y = y_fixed';
e = std(y_random,[],2)./sqrt(14); e = e';
pop.col = {[0.2,0.2,0.2]};
pop.width = 5;
h = mseb_no_edge(x,y,e,pop,0);
xlabel('GV');
ylabel('mean(rD) +/- SE(rD)');
% title('Linear Mixed Effect Model: Fixed and Random Effects');
title('LME: fixed effect & subject effect');
% legend(arrayfun(@(x) sprintf('Group %d', x), groupIDs, 'UniformOutput', false), 'Location', 'Best');
hold off;
set(gca,'FontSize',12)

nexttile
bar([groupSlope+fixedSlope])
hold on
% bar(groupSlope.*double((groupSlope_p<0.05)))
title('slope')
xlabel('subject')
ylabel('random + fixed slope')
set(gca,'FontSize',12)

nexttile

bar([groupOffset+fixedIntercept]')
title('intercept')
xlabel('subject')
ylabel('random + fixed intercept')
set(gca,'FontSize',12)


nexttile
x = X_pred';
y = y_fixed';
e = std(y_random,[],2)./sqrt(14); e = e';
pop.col = {[0.2,0.2,0.2]};
pop.width = 5;
h = mseb_no_edge(x,y,e,pop,0);
grid on
xticks([5:5:35])
xlim([0,35])
xlabel('GV');
ylabel('mean(rD) +/- SE(rD)');
legend('Averaged Fit across subjects')
title('LME: fixed effect line (shade for each subject)',['rD = ',num2str(fixedSlope),'*GV + ',num2str(fixedIntercept)]);
% legend(arrayfun(@(x) sprintf('Group %d', x), groupIDs, 'UniformOutput', false), 'Location', 'Best');
hold off;
set(gca,'FontSize',12)
%% Figure 3E/F: for each SNR, show subject curve first and do an average
% in 3D, we decided on "linear mixed effect line" to show this
% negative relationship so we still do this here with the same formula
% (allowing random slopes for each subject)

clearvars  -except data*

clc
formula = 'rD~EyeMovVel + (EyeMovVel|subj)';
disp(formula)
% data_used = data_fit;

data_used1 = data_tmp_GVsmooth(and(data_tmp_GVsmooth.EyeMovVel>0,data_tmp_GVsmooth.EyeMovVel<50),:);
data_used2 = data_GVsmooth_full(and(data_GVsmooth_full.EyeMovVel>0,data_GVsmooth_full.EyeMovVel<50),:);


xA_Name = 'SNR';
xA = data_used1.SNR;
[xA1,xA2] = discretize(xA,[-12:2:4]);
xA_bin = xA2(xA1);
clear xA1 xA2

xB = data_used1.EyeMovVel;
xB_Name = 'GV';%'Gaze Velocity (deg/s)';%Blink Rate';%'EyeMovVel';%'BR';%'EyeMovVel';

y_Name = 'rD';
Y = data_used1.rD;

% in the full range of SNR, fit lme
figure('Position',[312 44 443 664])
tiledlayout(8,1,'TileSpacing','tight')
nexttile([4,1])
xA_cnt  = 0 ;
cmap = [jet(numel(unique(xA_bin))+1)];

for xA_now = [unique(xA_bin),9999]
    xA_cnt = xA_cnt + 1;
    if xA_now == 9999
        % the full SI
        tmp_lme = fitlme(data_used2,formula, 'FitMethod', 'ML');
        x_pred = linspace(min(data_used2.EyeMovVel),max(data_used2.EyeMovVel),100)';

    else
        index1 = find(xA_bin == xA_now);
        tmp_lme = fitlme(data_used1(index1,:),formula, 'FitMethod', 'ML');

        % plot the fixed line +/- se from random lines
        x_pred = linspace(min(xB(index1)),max(xB(index1)),100)';
    end

    hold on
    % scatter(xB(index1),Y(index1),[],cmap(xA_cnt,:),'filled','MarkerFaceAlpha',0.2)
    indx = find(strcmpi(tmp_lme.CoefficientNames,'EyeMovVel'));
    Coeffs = tmp_lme.fixedEffects;
    fixedIntercept(xA_cnt)  = Coeffs(1);
    fixedSlope(xA_cnt)  = Coeffs(indx);
    Coeffs_p = tmp_lme.Coefficients.pValue;
    pval(xA_cnt) = Coeffs_p(indx);


    [randomIntercepts,~,randomIntercepts_table] = randomEffects(tmp_lme);
    randomPval(:,xA_cnt) = randomIntercepts_table.pValue;

    y_fixed(:,xA_cnt) = fixedIntercept(xA_cnt) + fixedSlope(xA_cnt) .* x_pred; % 固定效应拟合线

    sub_list = unique(data_used1.subj);
    for sub_cnt = 1:14
        clear indx*
        indx1 = find(strcmpi(randomIntercepts_table.Name,'(Intercept)'));
        indx2 = find(strcmpi(randomIntercepts_table.Level,num2str(sub_list(sub_cnt))));
        indx_intercept = intersect(indx1,indx2);

        clear indx1 indx2
        indx1 = find(strcmpi(randomIntercepts_table.Name,'EyeMovVel'));
        indx2 = find(strcmpi(randomIntercepts_table.Level,num2str(sub_list(sub_cnt))));
        indx_slope = intersect(indx1,indx2);

        if ~isempty(indx_intercept)
            groupOffset(sub_cnt) = randomIntercepts(indx_intercept); % 该组的随机截距

        end


        if ~isempty(indx_slope)
            groupSlope(sub_cnt) = randomIntercepts(indx_slope);
            y_random(:,sub_cnt) = (fixedIntercept(xA_cnt) + groupOffset(sub_cnt)) + (fixedSlope(xA_cnt) + groupSlope(sub_cnt)) .* x_pred; % 该组的拟合线

            allSlope(sub_cnt) = (fixedSlope(xA_cnt) + groupSlope(sub_cnt));
            allIntercept(sub_cnt) =  (fixedIntercept(xA_cnt) + groupOffset(sub_cnt));
        else

            y_random(:,sub_cnt) = (fixedIntercept(xA_cnt) + groupOffset(sub_cnt)) + (fixedSlope(xA_cnt)) .* x_pred; % 该组的拟合线

            allSlope(sub_cnt) = (fixedSlope(xA_cnt));
            allIntercept(sub_cnt) =  (fixedIntercept(xA_cnt) + groupOffset(sub_cnt));
        end






    end

    meanAllSlope(xA_cnt) = mean(allSlope);
    meanAllIntercept(xA_cnt) = mean(allIntercept);
    if xA_now~=10000

        hold on
        clear x y e pop h
        x = x_pred';
        y = y_fixed(:,xA_cnt)';
        e = std(y_random,[],2)./sqrt(14); e = e';
        pop.col = {cmap(xA_cnt,:)};
        pop.width = 2;
        h = mseb_no_edge(x,y,e,pop,0);
    else
        hold on
        clear x y e pop h
        x = x_pred';
        y = y_fixed(:,xA_cnt)';
        e = std(y_random,[],2)./sqrt(14); e = e';
        pop.col = {[0,0,0]};
        pop.width = 5;
        pop.stype = ':';
        h = mseb_no_edge(x,y,e,pop,0);
    end


end

hold on
% plot the average
x = linspace(min(xB),max(xB),100);
y = mean(y_fixed(:,1:end-1),2);
e = std(y_fixed(:,1:end-1),[],2)./sqrt(14); e = e';
pop.col = {[0,0,0]};
pop.width = 5;
h = mseb_no_edge(x,y,e,pop,0);
ylim([-0.02,0.15])
xlim([0,40]);

xlabel('GV');
ylabel('mean(rD) +/- SE(rD)');
% title('Linear Mixed Effect Model: Fixed and Random Effects');
title('LME: fixed effect & subject effect',['Average: ','rD ~ ',num2str(mean(meanAllSlope(1:end-1))),'*GV + ',num2str(mean(meanAllIntercept(1:end-1)))]);
% legend(arrayfun(@(x) sprintf('Group %d', x), groupIDs, 'UniformOutput', false), 'Location', 'Best');
hold off;
set(gca,'FontSize',12)
grid on
ylim([-0.1,0.25])


nexttile([2,1])
bar([-12:2:4,6],[meanAllSlope(1:end),mean(meanAllSlope(1:end));fixedSlope(1:end),mean(fixedSlope(1:end))])
legend({'Mean Slope (fixed + random)','Fixed Effect'},'Location','best')
hold on
% bar([-12:2:4],meanAllSlope.*double(pval<0.05),'DisplayName','Significant')
labels = num2cell([-12:2:4]);
labels{end} = 'full';
labels{end+1} = 'avg';

xticklabels(labels)
clear all_slope_ labels
ylabel('Slope');
title('Slope of GV across SNR',formula)

nexttile([2,1])
colormap(jet(10))
bar([-12:2:4,6]',[meanAllSlope(1:end),mean(meanAllSlope(1:end))]');
legend({'Mean Slope (fixed + random)','Fixed Effect'},'Location','best')
hold on
% bar([-12:2:4],meanAllSlope.*double(pval<0.05),'DisplayName','Significant')
labels = num2cell([-12:2:4]);
labels{end} = 'full';
labels{end+1} = 'avg';

xticklabels(labels)
clear all_slope_ labels
ylabel('Slope');
title('Slope of GV across SNR',formula)

%% Figure 4: LME model
clc
clearvars -except data*
figure;
tiledlayout('flow')

for datasetName = {'data_GVsmooth_not_full','data_GVsmooth_full'}
    clear data_used
    datasetName = datasetName{1};
    eval(['data_used =',datasetName,';' ])

    data_used.EyeMovVel = normalize(data_used.EyeMovVel,'norm');
    data_used.SNR = normalize(data_used.SNR,'norm');
    data_used.rINT = normalize(data_used.rINT,'norm');
    data_used.hitP = normalize(data_used.hitP,'norm');

    best_model = fitlme(data_used,'rD ~  EyeMovVel*SNR + rINT+hitP +  (1|subj)');

    % compare model to a null model (only random term)
    lme_null = fitlme(data_used, 'rD ~ 1 + (1|subj)');
    compare(lme_null,best_model)

    % get features
    fixedE = fixedEffects(best_model); % get fixed effect
    feNames = best_model.CoefficientNames;   % get var names

    % get 95% CI and p values
    feCI = coefCI(best_model);
    pValues = best_model.Coefficients.pValue;

    % get all slopes
    nonInterceptIdx = and(~strcmp(feNames, '(Intercept)'),~contains(feNames, ':'));
    fixedE = fixedE(nonInterceptIdx);
    feCI = feCI(nonInterceptIdx, :);
    feNames = feNames(nonInterceptIdx);
    pValues = pValues(nonInterceptIdx);

    y_positions = linspace(1, length(fixedE), length(fixedE));

    nexttile

    hold on
    for i = 1:length(fixedE)
        if pValues(i) < 0.05
            color = 'k';
        else
            color = [0.6 0.6 0.6];
        end


        plot([feCI(i,1), feCI(i,2)], [y_positions(i), y_positions(i)], '-', 'Color', color, 'LineWidth', 2);
        plot(fixedE(i), y_positions(i), 'o', 'MarkerFaceColor', color, 'MarkerEdgeColor', color, 'MarkerSize', 6);
    end

    hold off;

    yticks(y_positions);

    feNames{strcmpi(feNames,'EyeMovVel')} = 'GV';
    feNames{strcmpi(feNames,'rINT')} = 'SI';
    try
        feNames{strcmpi(feNames,'hitP')} = 'HR';
    catch
    end
    yticklabels(feNames);
    xlabel('Fixed Effect Estimate');
    ylabel('Predictor');
    if contains(datasetName,'not')
        datasetName = 'SI not full';
    else
        datasetName = 'full SI';
    end
    title('Main Effects with 95% Confidence Intervals',datasetName);

    % format
    set(gca, 'YDir', 'reverse');
    grid on;
    xline(0, '--', 'Color', [0.5 0.5 0.5]);
    ylim([min(y_positions)-0.5, max(y_positions)+0.5]);
    set(gca, 'FontSize', 12);

    clear feNames feCI fixedE feCI nonInterceptIdx pValues



end

%% Figure 4C1: Effect of SNR on SI
% fit LMEs
clearvars -except data*
data_used = data_tmp_GVsmooth;
data_used.SNR = normalize(data_used.SNR,'norm');
data_used.rINT = normalize(data_used.rINT,'norm');
data_used.EyeMovVel = normalize(data_used.EyeMovVel,'norm');

lm = fitlme(data_used, 'rINT ~ SNR + (SNR|subj)');

% get fix and random effects
fixed_effects = lm.Coefficients.Estimate;  % fixed effect
[~,~,random_effects] = lm.randomEffects;  % random effect
subjects = unique(data_fit.subj);  % all subjects
n_subjects = numel(subjects);

% compute the composite effect (fixed + random)
index = strcmpi(random_effects.Name,'SNR');
composite_effect = fixed_effects(2) + random_effects.Estimate(index);

ci = coefCI(lm);  % CI for the fixed effect
composite_ci = [composite_effect - 1.96 * random_effects.SEPred(index), composite_effect + 1.96 * random_effects.SEPred(index)];  % 综合效应的CI


[sorted_subjects, idx] = sort(subjects, 'ascend');
sorted_composite_effect = composite_effect(idx);
sorted_composite_ci = composite_ci(idx, :);


% forest figure
figure;
hold on;
cmap = lines;

% fixed effect and CI
scatter(fixed_effects(2), 0, [],60,'filled','MarkerFaceColor',cmap(1,:));
plot([ci(2,1) ci(2,2)], [0, 0], '-', 'LineWidth', 1.5,'Color',cmap(1,:));
p_value = lm.Coefficients.pValue(2);
if p_value < 0.01
    text(fixed_effects(2), 0, '  ***', 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
elseif p_value < 0.05
    text(fixed_effects(2), 0, '  *', 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
end

% each subject's random effect
gap1 = 0.25;
gap2 = 0.125;
for i = 1:n_subjects
    hold on
    scatter(sorted_composite_effect(i),gap1+i*gap2,30,cmap(1,:),'filled');

    if sorted_composite_ci(i,1).*sorted_composite_ci(i,2)>0 % same sign
        plot([sorted_composite_ci(i,1) sorted_composite_ci(i,2)], gap1+[i*gap2, i*gap2], 'Color', cmap(1,:), 'LineStyle', '--', 'LineWidth', 1);

    else
        plot([sorted_composite_ci(i,1) sorted_composite_ci(i,2)], gap1+[i*gap2, i*gap2], 'Color', cmap(1,:), 'LineStyle', '--', 'LineWidth', 0.5);
    end
end

% visualization
set(gca, 'YTick', gap1+[1:n_subjects].*gap2, 'YTickLabel', [1:14]);
set(gca,'YDir','reverse');
xlabel('Effect Size');
ylabel('Subjects');
title('Forest Plot for SI ~ SNR + (SNR|subj)');


legend({'Fixed Effect (mean)', 'CI (Fixed Effect)', 'Subject-Specific Effects', 'CI (Subject-Specific)'});

hold off;


%% Figure 4C2: Effect of SNR on GV
figure;
clearvars -except data* cmap
hold on
lm = fitlme(data_used, 'EyeMovVel ~ SNR + (SNR|subj)');


fixed_effects = lm.Coefficients.Estimate;  % fixed effect
[~,~,random_effects] = lm.randomEffects;  % random effects
subjects = unique(data_fit.subj);  % all subjects
n_subjects = numel(subjects);

% compute the composite effect
index = strcmpi(random_effects.Name,'SNR');
composite_effect = fixed_effects(2) + random_effects.Estimate(index);
ci = coefCI(lm);  % CI for the fixed effect
composite_ci = [composite_effect - 1.96 * random_effects.SEPred(index), composite_effect + 1.96 * random_effects.SEPred(index)];  % 综合效应的CI


[sorted_subjects, idx] = sort(subjects, 'ascend');
sorted_composite_effect = composite_effect(idx);
sorted_composite_ci = composite_ci(idx, :);


% fixed effect and CI and p val
scatter(fixed_effects(2), 0, [],60,'filled','MarkerFaceColor',cmap(2,:));
plot([ci(2,1) ci(2,2)], [0, 0], '-', 'LineWidth', 1.5,'Color',cmap(2,:));

p_value = lm.Coefficients.pValue(2);
if p_value < 0.01
    text(fixed_effects(2), 0, '  ***', 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
elseif p_value < 0.05
    text(fixed_effects(2), 0, '  *', 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
end


% each subject's random effect
gap1 = 0.25;
gap2 = 0.125;
for i = 1:n_subjects
    hold on
    scatter(sorted_composite_effect(i),gap1+i*gap2,30,cmap(2,:),'filled');


    % signfiicance based on CI
    if sorted_composite_ci(i,1).*sorted_composite_ci(i,2)>0 % same sign
        plot([sorted_composite_ci(i,1) sorted_composite_ci(i,2)], gap1+[i*gap2, i*gap2], 'Color', cmap(2,:), 'LineStyle', '--', 'LineWidth', 1);

    else
        plot([sorted_composite_ci(i,1) sorted_composite_ci(i,2)], gap1+[i*gap2, i*gap2], 'Color', cmap(2,:), 'LineStyle', '--', 'LineWidth', 0.5);
    end

end

% visualization
set(gca, 'YTick', [0,gap1+[1:n_subjects].*gap2], 'YTickLabel', ['Fixed Effect',num2cell([1:14])]);
set(gca,'YDir','reverse');
xlabel('Effect Size');
ylabel('Subjects');
legend({'Fixed Effect (mean)', 'CI (Fixed Effect)', 'Subject-Specific Effects', 'CI (Subject-Specific)'});
hold off;
xline(0,'k--');
ylim([-0.25,2.25])

%% Figure 4D: interaction (SI vs SNR vs rD done)

clearvars -except data*
clear data_used
data_used = data_fit;
x_vals_global = linspace(min(data_used.rINT), max(data_used.rINT), 10)';
x_names = 'SI';
group_names = 'SNR';
group_levels = [-12:2:4];


subj_list = unique(data_used.subj);

figure; hold on;
cmap = hot;
cmap = cmap([10:171],:);
colors = downsample(cmap,round(size(cmap,1)/numel(group_levels)));


mean_rd = nan(numel(group_levels), 10);
se_rd = nan(numel(group_levels), 10);

for i = 1:numel(group_levels)
    all_preds = nan(numel(subj_list), 10);  % save predicted vals

    for j = 1:numel(subj_list)
        data_used_curr = data_used(data_used.subj==subj_list(j),:);
        lm1 =  fitlm(data_used_curr, 'rD ~ EyeMovVel*rINT*SNR');
        x_vals = x_vals_global;
        % predict for each subject with fixed terms
        pred_table = table(x_vals(:), ...               % x var
            repmat(group_levels(i), numel(x_vals), 1), ... %  group
            mean(data_used_curr.EyeMovVel) * ones(numel(x_vals), 1), ... % avg others
            mean(data_used_curr.hitP) * ones(numel(x_vals), 1), ... % avg others
            repmat(subj_list(j), numel(x_vals), 1), ... % each subject
            'VariableNames', {'rINT', 'SNR', 'EyeMovVel', 'hitP','subj'});


        all_preds(j, :) = predict(lm1, pred_table);
        clear   pred_table lm1 data_used_curr
    end


    slopes_si_subjects = diff(all_preds,[],2)./diff(x_vals_global');
    slopes_si_subjects = slopes_si_subjects(:,1);
    slopes_groups(i,:) = slopes_si_subjects';
    clear slopes_si_subjects

    mean_rd(i, :) = mean(all_preds, 1);
    se_rd(i, :) = std(all_preds, [], 1) / sqrt(numel(subj_list));

    % mean
    plot(x_vals_global, mean_rd(i, :), '-', 'Color', colors(i,:), 'LineWidth',2,...
        'DisplayName', [group_names,' = ' num2str(group_levels(i))]);

end

legend show;
xlabel(x_names);
ylabel('Predicted rD');
title(['Modeling: rD vs ', x_names, ' (SNR fixed)' ])
grid on;
hold off;

set(gca,'FontSize',12)

% get the slope for each SNR:
slopes_si = diff(mean_rd,[],2)./diff(x_vals_global);
slopes_si = slopes_si(:,1);
slopes_si_subjects = slopes_groups;


clearvars -except data* slopes_si*
data_used = data_fit;
x_vals_global = linspace(min(data_used.EyeMovVel), max(data_used.EyeMovVel), 10)';  % X 变化范围
x_names = 'GV';
group_names = 'SNR';
group_levels = [-12:2:4];


subj_list = unique(data_used.subj);

figure; hold on;
cmap = hot;
cmap = cmap([10:171],:);
colors = downsample(cmap,round(size(cmap,1)/numel(group_levels)));


mean_rd = nan(numel(group_levels), 10);
se_rd = nan(numel(group_levels), 10);

for i = 1:numel(group_levels)
    all_preds = nan(numel(subj_list), 10);  % 每个 subject 的预测值

    for j = 1:numel(subj_list)
        data_used_curr = data_used(data_used.subj==subj_list(j),:);
        lm1 =  fitlm(data_used_curr, 'rD ~ EyeMovVel*rINT*SNR');
        x_vals = x_vals_global;

        pred_table = table(x_vals(:), ...               % x var
            repmat(group_levels(i), numel(x_vals), 1), ... % each group
            mean(data_used_curr.rINT) * ones(numel(x_vals), 1), ... % avg others
            mean(data_used_curr.hitP) * ones(numel(x_vals), 1), ... % avg others
            repmat(subj_list(j), numel(x_vals), 1), ... % each subject
            'VariableNames', {'EyeMovVel', 'SNR', 'rINT', 'hitP','subj'});

        % predict rD
        all_preds(j, :) = predict(lm1, pred_table);
        clear   pred_table lm1 data_used_curr
    end

    slopes_gv_subjects = diff(all_preds,[],2)./diff(x_vals_global');
    slopes_gv_subjects = slopes_gv_subjects(:,1);
    slopes_groups(i,:) = slopes_gv_subjects';
    clear slopes_gv_subjects

    % mean and se
    mean_rd(i, :) = mean(all_preds, 1);
    se_rd(i, :) = std(all_preds, [], 1) / sqrt(numel(subj_list)); % 计算标准误

    % plot avg
    plot(x_vals_global, mean_rd(i, :), '-', 'Color', colors(i,:), 'LineWidth',2,...
        'DisplayName', [group_names,' = ' num2str(group_levels(i))]);


end

legend show;
xlabel(x_names);
ylabel('Predicted rD');
title(['Modeling: rD vs ', x_names, ' (SNR fixed)' ])
grid on;
hold off;

set(gca,'FontSize',12)
% get the slope for each SNR:
slopes_gv = diff(mean_rd,[],2)./diff(x_vals_global);
slopes_gv = slopes_gv(:,1);
% slopes_gv_subjects = diff(all_preds,[],2)./diff(x_vals_global');
slopes_gv_subjects = slopes_groups;

figure;
tiledlayout('flow')
nexttile
% plot bar plot (slopes of SI and GV across SNRs)
group_names = 'SNR';
group_levels = [-12:2:4]';%[0:10:100];
bar(group_levels,slopes_si,'EdgeColor','none','DisplayName','slopes(SI)','BarWidth',0.6);
hold on
bar(group_levels,slopes_gv,'EdgeColor','none','DisplayName','slopes(GV)','BarWidth',0.6);
ylim(max(abs([slopes_si;slopes_gv])).*[-1.1,1.1]);
xlabel('SNR');
ylabel('Slopes')
title('Subject-averaged slopes between SI(GV) and rD across SNR levels')
set(gca,'FontSize',12);


%% appendix: functions
function tstats = return_tstats(x)
[~,~,~,tstats] = ttest(x,0,'Tail','right');
tstats = tstats.tstat;

end

function p = return_ttest_p(x)
[~,p,~,~] = ttest(x,0,'Tail','right');

end

function p = return_corr_p(x,y)
[~,p] = corr(x,y);

end

function [XX,YY,EE,stds] =find_average_line(x,y)
% x is the value for grouping
% y is the value to be averaged
[x,y] = prepareCurveData(x,y);
%
%     val = sortrows([x,y]);
%     x = val(:,1);
%     y = val(:,2);
%
%     numbers = hist(x,unique(x)); % count number for each unique element

[grouping,id_x] = findgroups(x);

% for all data samples
means = splitapply(@mean,y,grouping);
nums = splitapply(@numel,y,grouping);
stes = splitapply(@(x)std(x)/sqrt(numel(x)),y,grouping);
stds = splitapply(@std,y,grouping);

XX = id_x; %=unique(x)
YY = means;
EE = stes;


%     XX = XX(nums>=14);
%     YY = YY(nums>=14);

end
