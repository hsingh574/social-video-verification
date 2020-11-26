function scores = compute_stats(x)
% Compute statistics for each of the samples in x
% where the sample axis is the last axis

warning('off','all');

%output = zeros(

for c = 1:6
    temp = squeeze(x(c,:,:));
    for f = 1:size(temp,1)
        temp2 = transpose(temp(f,:));
	%temp2 is of shape (51,1)








warning('on','all');
end 
