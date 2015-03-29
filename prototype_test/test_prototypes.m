addpath('../');
create_data(1000, 10, 10);
compute_kernels();
load('data.mat', 'y');
load('kernels.mat', 'K_all');
[alphas_sub, stats_sub] = subgradient(y,K_all,1, 0.01, 5);
[alphas_bcd, stats_bcd] = bcd_exact(y,K_all,1, 0.01, 5);
opt_obj = stats_bcd.objective(end);
figure('name', 'f vs time');
semilogy(stats_sub.time,stats_sub.objective-opt_obj,'r-+');
semilogy(stats_bcd.time,stats_bcd.objective-opt_obj,'g-o');
legend('subgradient', 'BCD-exact');



