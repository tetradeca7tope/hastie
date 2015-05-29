function K = kernel(s, t)

% s and t are vertical vectors with length m and n. K is the m by n matrix:  K(s_i, t_j) = k_1(s_i) * k_1(t_j) + k_2(s_i)*k2(t_j) - k4(abs(s_i - t_j)). see the paper for k1, k2, k4.

m = length(s);
n = length(t);
b = abs(s * ones(1,n) - ones(m, 1) * (transpose(t)));
k1s = s - 0.5;
k1t = t - 0.5;
k2s = (k1s.^2 - 1/12)/2;
k2t = (k1t.^2 - 1/12)/2;
K = k1s * k1t' + k2s * k2t' - ((b - 0.5).^4 - (b - 0.5).^2 / 2 + 7/240)/24;
