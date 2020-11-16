function scores = compute_stats(x)
% Compute statistics for each of the samples in x
% where the sample axis is the last axis

warning('off','all');
[nrows, dcols] = size(x);
scores = zeros(nrows,1);
nsample = ceil( nrows*0.95);
 if (nsample > 300)
  nsample = 300;
 end
for i = 1:nrows
 %#sample points to build dpop
 tmp1 = randperm(nrows); tmp1 = tmp1(1:nsample); dpop = x(tmp1, : );
 distSample0 = zeros(nsample,1);

 for j = 1:nsample %#build distances from point i to sampled points
  distSample0(j,1) = sqrt(sum((dpop(j,:) - x(i,:)).^2));
 end
 tmp2 = randperm(nrows); tmp2 = tmp2(1:nsample); bpop = x(tmp2, : );
 for k = 1:nsample  %#build distances from bpop k to sampled points dpop
  distSampleTemp = zeros(nsample,1);
   for j = 1:nsample
    distSampleTemp(j,1) = sqrt(sum(( dpop(j,:) -  bpop(k,:)).^2));
   end
  [pval, ks, d] = kstest2(distSample0,distSampleTemp);
  scores(i,1) = scores(i,1) + d/nsample;
 end

end
warning('on','all');
end 
