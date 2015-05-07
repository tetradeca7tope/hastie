load('october_traits.mat');
traitIndividualIDs = individualIDs; clear individualIDs;
load('october_snps.mat');
markerIndividualIDs = individualIDs; clear individualIDs;
assert(isequal(traitIndividualIDs,markerIndividualIDs));

geno_101 = geno - 1;
geno_scores = PCAscores(geno_101, 0);
meta.insulin_name = 'Insulin_RBM_log_sex_W10';
insulin_idx = find(strcmp(traitNames, meta.insulin_name));
sorted_scores = sort(geno_scores,'descend');
top100_scores = find(geno_scores >= sorted_scores(50));

% Make sure all chromosomes represented in reduced set:
%assert(length(unique(markerChromosome(top100_scores))) == 20);

insulin_data = traitData(:,insulin_idx);
snp_data = geno_101(:,top100_scores);
meta.snp_ids = {markerIDs{top100_scores}}';
meta.snp_chromosomes = markerChromosome(top100_scores);
meta.individual_ids = traitIndividualIDs;
meta.snp_positions = markerPosition(top100_scores);
save('epidata.mat', 'insulin_data', 'snp_data', 'meta');
