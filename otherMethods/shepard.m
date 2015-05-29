function [Yt] = shepard(X, Y, c, Xt)
% SHEPARD
% Simple Shepard interpolation
%
% Input
% X, Y   : Data points (X(i,:), Y(i)), i = 1, ..., n
% c      : Parameter c in the power-law function (typically equal to 2 or
%          somewhere between 1 and 3). Greater values of c assign greater
%          influence to values closest to the interpolated point. For
%          0 < c < 1 interpolation has smooth peaks over the interpolated
%          points, while as c > 1 the peaks become sharp. The choice of
%          value for c is therefore a function of the degree of smoothing
%          desired in the interpolation, the density and distribution of
%          samples being interpolated, and the maximum distance over which
%          an individual sample is allowed to influence the surrounding
%          ones.
% Xt     : Input values of the test data points (responses of which must be
%          predicted)
%
% Output
% Yt     : Predicted responses at Xt

% This source code is tested with Matlab version 7.1 (R14).

% =========================================================================
% Simple Shepard interpolation
% Last changes: December 14, 2009
% Author: Gints Jekabsons (gints.jekabsons@rtu.lv)
% URL: http://www.cs.rtu.lv/jekabsons/
%
% Copyright (c) 2009  Gints Jekabsons
%
% This software is free for any use.
%
% THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
% NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
% DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
% OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
% USE OR OTHER DEALINGS IN THE SOFTWARE.
% =========================================================================

if nargin < 4
    error('Too few input arguments.');
end
[n, d] = size(X);
if n ~= size(Y,1)
    error('X and Y should have the same number of rows.')
end
if d ~= size(Xt,2)
    error('X and Xt should have the same number of columns.')
end

nt = size(Xt,1);
for k = 1 : nt
    SumA = 0;
    SumW = 0;
    for i = 1 : n
        EuclideanDist = norm(X(i,:) - Xt(k,:));
        if EuclideanDist == 0  % the case when there is exactly such data point in training data
            Yt(k,1) = Y(i,1);
            break
        end
        Pow = EuclideanDist ^ -c;
        SumA = SumA + Pow;
        SumW = SumW + Pow * Y(i,1);
    end
    if EuclideanDist ~= 0  
        Yt(k,1) = SumW / SumA;
    end
end
