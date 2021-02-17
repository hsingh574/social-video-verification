% This script was used to generate most of the plots in the paper. 
%
% Written by Eleanor Tursman
% Last updated 11/2020 (Harman Suri)

%% Make the plots
clearvars; close all;

accOn = true;
prOn = false;
rocOn = true;

%% make accuracy plots
% (1) TP if window contains a faked frame & fake is detected
% (2) TN if window does not have fake & fake is not detected
% (3) FP if window does not have fake & fake is detected
% (4) FN if window contains a faked frame & fake is not detected
% Accuracy = (TP+TN)/(TP+TN+FP+FN)
if (accOn)
    
    people = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'};
    accResults = zeros(4,4,length(people));
    thresh = 1; % ind in [1.3 1.5 1.7 1.9 2.1], Note 0-indexing here 
    
    for p=1:length(people)
        
	if p==17
		continue	
	end

        fnameRoot = ['results_v2/ID' num2str(p) '/thresh_' num2str(thresh) '/'];
        
        % load the data
        win50 = load([fnameRoot 'window_50.mat']);
        win150 = load([fnameRoot 'window_150.mat']);
        win250 = load([fnameRoot 'window_250.mat']);
        win350 = load([fnameRoot 'window_350.mat']);
        
        dataset = {win50,win150,win250,win350};
        accXData = [50 150 250 350];
        
        numWin = length(dataset);
        accs = zeros(4,numWin);
        
        for i=1:numWin
            
            data = dataset{i};
            accs(1,i) = (sum(data.acc0(1,:)) + sum(data.acc0(2,:))) / (sum(data.acc0(1,:)) + sum(data.acc0(2,:)) + sum(data.acc0(3,:)) + sum(data.acc0(4,:)));
            accs(2,i) = (sum(data.acc1(1,:)) + sum(data.acc1(2,:))) / (sum(data.acc1(1,:)) + sum(data.acc1(2,:)) + sum(data.acc1(3,:)) + sum(data.acc1(4,:)));
            accs(3,i) = (sum(data.acc2(1,:)) + sum(data.acc2(2,:))) / (sum(data.acc2(1,:)) + sum(data.acc2(2,:)) + sum(data.acc2(3,:)) + sum(data.acc2(4,:)));
            accs(4,i) = (sum(data.acc3(1,:)) + sum(data.acc3(2,:))) / (sum(data.acc3(1,:)) + sum(data.acc3(2,:)) + sum(data.acc3(3,:)) + sum(data.acc3(4,:)));
            
        end
        
        accResults(:,:,p) = accs;
        
    end
    
    
    % Plot average result per window + one stdev
    meanRes = mean(accResults,3);
    stdRes = std(accResults,0,3);
   
    disp(meanRes)

    figure;
    errorbar(accXData,meanRes(1,:),stdRes(1,:)); hold on;
    errorbar(accXData,meanRes(2,:),stdRes(2,:)); hold on;
    errorbar(accXData,meanRes(3,:),stdRes(3,:)); hold on;
    errorbar(accXData,meanRes(4,:),stdRes(4,:)); hold on;
    ylim([0 1]);
    xlim([0 400]);
    xlabel('Window Size');
    ylabel('Accuracy');
    title('Detection Accuracy vs Window Size');
    set(gca,'FontSize',20);
    legend('No Fakes','One Fake','Two Fakes','Three Fakes','Location','SouthEast');
    savefig('acc_full_check.fig')
end

%% precision recall
% (1) TP if window contains a faked frame & fake is detected
% (2) TN if window does not have fake & fake is not detected
% (3) FP if window does not have fake & fake is detected
% (4) FN if window contains a faked frame & fake is not detected
% precision = tp / (tp + fp), recall = tp / (tp + fn)

if(prOn)
    
    people = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'};
    threshNum = 6; % length of [1.1 1.3 1.5 1.7 1.9 2.1]
    win = 250; % [50 150 250 350]
    pResults = zeros(threshNum,3,length(people));
    rResults = zeros(threshNum,3,length(people));
    
    for p=1:length(people)
        
	if p==17
		continue
	end

        fnameRoot = ['Output/ID' num2str(p) '/'];
        
        data1 = load([fnameRoot 'thresh_1/' datasetName '_window_' num2str(win) '.mat']);
        data2 = load([fnameRoot 'thresh_2/' datasetName '_window_' num2str(win) '.mat']);
        data3 = load([fnameRoot 'thresh_3/' datasetName '_window_' num2str(win) '.mat']);
        data4 = load([fnameRoot 'thresh_4/' datasetName '_window_' num2str(win) '.mat']);
        data5 = load([fnameRoot 'thresh_5/' datasetName '_window_' num2str(win) '.mat']);
        data6 = load([fnameRoot 'thresh_6/' datasetName '_window_' num2str(win) '.mat']);
        
        dataset = {data1,data2,data3,data4,data5,data6};
        
        for i=1:threshNum
            
            data = dataset{i};
            
            if (sum(data.acc1(1,:)) == 0) && ( (sum(data.acc1(1,:)) + sum(data.acc1(3,:))) == 0)
                pResults(i,1,p) = 1;
            else
                pResults(i,1,p) = sum(data.acc1(1,:)) / (sum(data.acc1(1,:)) + sum(data.acc1(3,:)));
            end
            
            if (sum(data.acc2(1,:)) == 0) && ( (sum(data.acc2(1,:)) + sum(data.acc2(3,:))) == 0)
                pResults(i,2,p) = 1;
            else
                pResults(i,2,p) = sum(data.acc2(1,:)) / (sum(data.acc2(1,:)) + sum(data.acc2(3,:)));
            end
            
            % Running into NaN errors for 0/0
            if (sum(data.acc3(1,:)) == 0) && ( (sum(data.acc3(1,:)) + sum(data.acc3(3,:))) == 0)
                pResults(i,3,p) = 1;
            else
                pResults(i,3,p) = sum(data.acc3(1,:)) / (sum(data.acc3(1,:)) + sum(data.acc3(3,:)));
            end
            
            rResults(i,1,p) = sum(data.acc1(1,:)) / (sum(data.acc1(1,:)) + sum(data.acc1(4,:)));
            rResults(i,2,p) = sum(data.acc2(1,:)) / (sum(data.acc2(1,:)) + sum(data.acc2(4,:)));
            rResults(i,3,p) = sum(data.acc3(1,:)) / (sum(data.acc3(1,:)) + sum(data.acc3(4,:)));
            
        end
        
    end
    
    % Aggregate results-- can't do std without taking multiple measurements
    % of -each- threshold
    meanP = mean(pResults,3);
    meanR = mean(rResults,3);
    stdP = std(pResults,0,3);
    stdR = std(rResults,0,3);
    
    % reformat & order by recall values
    oneFake = sortrows([meanR(:,1) meanP(:,1)]);
    twoFake = sortrows([meanR(:,2) meanP(:,2)]);
    thrFake = sortrows([meanR(:,3) meanP(:,3)]);
    
    figure;
    errorbar(oneFake(:,1),oneFake(:,2),stdP(:,1),stdP(:,1),stdR(:,1),stdR(:,1)); hold on;
    errorbar(twoFake(:,1),twoFake(:,2),stdP(:,2),stdP(:,2),stdR(:,2),stdR(:,2)); hold on;
    errorbar(thrFake(:,1),thrFake(:,2),stdP(:,3),stdP(:,3),stdR(:,3),stdR(:,3)); hold on;
    legend('One Fake','Two Fakes','Three Fakes');
    xlabel('Recall');
    ylabel('Precision');
    xlim([0 1]);
    ylim([0 1]);
    title('Precision v Recall, Window size = 250');
    
end

%% make roc curves
% (1) TP if window contains a faked frame & fake is detected
% (2) TN if window does not have fake & fake is not detected
% (3) FP if window does not have fake & fake is detected
% (4) FN if window contains a faked frame & fake is not detected
% roc: TPR vs FPR at various threshes
% TPR = tp / (tp + fn)
% FPR = fp / (fp + tn)
if (rocOn)
    
    people = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'};
    threshNum = 5; % length of [1.3 1.5 1.7 1.9 2.1]
    win = 250; % [50 150 250 350]
    tpResults = zeros(threshNum,3,length(people));
    fpResults = zeros(threshNum,3,length(people));
    fpZeroFake = zeros(threshNum,1,length(people));
    
    for p=1:length(people)
	
	if p==17
		continue
	end
		    
        fnameRoot = ['results_v2/ID' num2str(p) '/'];
        
        data1 = load([fnameRoot 'thresh_0/' 'window_' num2str(win) '.mat']);
        data2 = load([fnameRoot 'thresh_1/' 'window_' num2str(win) '.mat']);
        data3 = load([fnameRoot 'thresh_2/' 'window_' num2str(win) '.mat']);
        data4 = load([fnameRoot 'thresh_3/' 'window_' num2str(win) '.mat']);
        data5 = load([fnameRoot 'thresh_4/' 'window_' num2str(win) '.mat']);
        
        dataset = {data1,data2,data3,data4,data5};
        
        for i=1:threshNum
            
            data = dataset{i};
            tpResults(i,1,p) = sum(data.acc1(1,:)) / (sum(data.acc1(1,:)) + sum(data.acc1(4,:)));
            tpResults(i,2,p) = sum(data.acc2(1,:)) / (sum(data.acc2(1,:)) + sum(data.acc2(4,:)));
            tpResults(i,3,p) = sum(data.acc3(1,:)) / (sum(data.acc3(1,:)) + sum(data.acc3(4,:)));
            
            fpResults(i,1,p) = sum(data.acc1(3,:)) / (sum(data.acc1(3,:)) + sum(data.acc1(2,:)));
            fpResults(i,2,p) = sum(data.acc2(3,:)) / (sum(data.acc2(3,:)) + sum(data.acc2(2,:)));
            fpResults(i,3,p) = sum(data.acc3(3,:)) / (sum(data.acc3(3,:)) + sum(data.acc3(2,:)));
            
            fpZeroFake(i,1,p) = sum(data.acc0(3,:)) / (sum(data.acc0(3,:)) + sum(data.acc0(2,:)));
            
        end
    end
    
    meanTP = mean(tpResults,3);
    meanFP = mean(fpResults,3)
    stdTP = std(tpResults,0,3);
    stdFP = std(fpResults,0,3)
    
    zeroFakeMean = mean(fpZeroFake,3)
    zeroFakeStd = std(fpZeroFake,0,3)
    
    % reformat & order by recall values
    oneFake = sortrows([meanFP(:,1) meanTP(:,1)]);
    twoFake = sortrows([meanFP(:,2) meanTP(:,2)]);
    thrFake = sortrows([meanFP(:,3) meanTP(:,3)]);
    
    disp(size(meanFP(:,1)))
    disp(size(oneFake))

    figure;
    errorbar(oneFake(:,1),oneFake(:,2),stdTP(:,1),stdTP(:,1),stdFP(:,1),stdFP(:,1)); hold on;
    errorbar(twoFake(:,1),twoFake(:,2),stdTP(:,2),stdTP(:,2),stdFP(:,2),stdFP(:,2)); hold on;
    errorbar(thrFake(:,1),thrFake(:,2),stdTP(:,3),stdTP(:,3),stdFP(:,3),stdFP(:,3)); hold on;
    legend('One Fake','Two Fakes','Three Fakes');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    xlim([0 1]);
    ylim([0 1]);
    title('ROC Curve, Window size = 250');
    set(gca,'FontSize',20);
    savefig('roc_full_check.fig') 
end
