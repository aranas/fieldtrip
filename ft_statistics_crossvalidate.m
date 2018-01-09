function [stat, cfg] = ft_statistics_crossvalidate(cfg, dat, design)

% FT_STATISTICS_CROSSVALIDATE performs cross-validation using a prespecified
% multivariate analysis given by cfg.mva
%
% Use as
%   stat = ft_timelockstatistics(cfg, data1, data2, data3, ...)
%   stat = ft_freqstatistics    (cfg, data1, data2, data3, ...)
%   stat = ft_sourcestatistics  (cfg, data1, data2, data3, ...)
%
% Options:
%   cfg.mva           = a multivariate analysis (default = {dml.standardizer dml.svm}) or string with user-specified function name
%   cfg.statistic     = a cell-array of statistics to report (default = {'accuracy' 'binomial'}); or string with user-specified function.
%   cfg.type          = a string specifying cross-validation scheme (default = nfold) /'nfold','split','loo','bloo';
%   cfg.nfolds        = number of cross-validation folds (default = 5)
%   cfg.resample      = true/false; upsample less occurring classes during
%                       training and downsample often occurring classes
%                       during testing (default = false)
%
% Returns:
%   stat.statistic    = the statistics to report
%   stat.model        = the models associated with this multivariate analysis
%
% See also FT_TIMELOCKSTATISTICS, FT_FREQSTATISTICS, FT_SOURCESTATISTICS

% Copyright (c) 2007-2011, F.C. Donders Centre, Marcel van Gerven
%
% This file is part of FieldTrip, see http://www.fieldtriptoolbox.org
% for the documentation and details.
%
%    FieldTrip is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    FieldTrip is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with FieldTrip. If not, see <http://www.gnu.org/licenses/>.
%
% $Id$


cfg.mva       = ft_getopt(cfg, 'mva');
cfg.statistic = ft_getopt(cfg, 'statistic', {'accuracy', 'binomial'});
cfg.nfolds    = ft_getopt(cfg, 'nfolds',   5);
cfg.resample  = ft_getopt(cfg, 'resample', false);
cfg.type      = ft_getopt(cfg, 'type','nfold');

% specify classification procedure or ensure it's the correct object
if isempty(cfg.mva)
    cfg.mva = dml.analysis({ dml.standardizer('verbose',true) ...
        dml.svm('verbose',true)});
elseif ~isa(cfg.mva,'dml.analysis')
    cfg.mva = dml.analysis(cfg.mva);
end

cv = dml.crossvalidator('mva', cfg.mva, 'type', cfg.type, 'folds', cfg.nfolds,...
    'resample', cfg.resample, 'compact', true, 'verbose', true);

if any(isinf(dat(:)))
    warning('Inf encountered; replacing by zeros');
    dat(isinf(dat(:))) = 0;
end

if any(isnan(dat(:)))
    warning('Nan encountered; replacing by zeros');
    dat(isnan(dat(:))) = 0;
end


if ischar(cfg.mva.method{1})
    mvafun = str2fun(cfg.mva.method{1});
    fprintf('using "%s" for crossvalidation\n', cfg.mva.method{1});
    
    X = dat';
    Y = design';
    
    % complete the folds when only train or test is specified
    if isempty(cv.trainfolds) && isempty(cv.testfolds)
        [cv.trainfolds,cv.testfolds] = cv.create_folds(Y);
    elseif isempty(cv.trainfolds)
        cv.trainfolds = cv.complement(Y,cv.testfolds);
    else
        cv.testfolds = cv.complement(Y,cv.trainfolds);
    end
    
    if iscell(Y)
        ndata = length(Y);
    else
        ndata = 1;
    end
    
    if ndata == 1
        nfolds = length(cv.trainfolds);
    else
        nfolds = length(cv.trainfolds{1});
    end
    
    cv.result = cell(nfolds,1);
    cv.design = cell(nfolds,1);
    cv.model = cell(nfolds,1);
    
    for f=1:nfolds % iterate over folds
        
        if cv.verbose
            if ndata == 1
                fprintf('validating fold %d of %d for %d datasets\n',f,nfolds,ndata);
            else
                fprintf('validating fold %d of %d for %d datasets\n',f,nfolds,ndata);
            end
        end
        
        % construct X and Y for each fold
        if ndata == 1
            trainX = X(cv.trainfolds{f},:);
            testX = X(cv.testfolds{f},:);
            trainY = Y(cv.trainfolds{f},:);
            testY = Y(cv.testfolds{f},:);
        else
            trainX = cell([length(X) 1]);
            testX = cell([length(X) 1]);
            for cv=1:length(X)
                trainX{cv} = X{cv}(cv.trainfolds{cv}{f},:);
                testX{cv} = X{cv}(cv.testfolds{cv}{f},:);
            end
            trainY = cell([length(Y) 1]);
            testY = cell([length(Y) 1]);
            for cv=1:length(Y)
                trainY{cv} = Y{cv}(cv.trainfolds{cv}{f},:);
                testY{cv} = Y{cv}(cv.testfolds{cv}{f},:);
            end
        end
        
        nout                        = nargout(mvafun);
        outputs                     = cell(1, nout-2);
        
        [model,result,outputs{:}]   = mvafun(cfg,trainX,testX,trainY);
        cv.model{f}.weights         = model;
        cv.result{f}                = result;
        cv.design{f}                = testY;
        out{f}                      = outputs{:};
        
        
        clear varargout;
        clear model;
        clear result;
        clear trainX;
        clear testX;
        clear trainY;
        clear testY;
        
    end
    
    % return unique model instead of cell array in case of one fold
    if length(cv.model)==1, cv.model = cv.model{1}; end
    
else
    fprintf('using DMLT toolbox\n');
    % perform everything
    cv = cv.train(dat',design');
end


% extract the statistic of interest
try
s = cv.statistic(cfg.statistic);
for i=1:length(cfg.statistic)
    stat.statistic.(cfg.statistic{i}) = s{i};
end
% if defined statistic is not part of toolbox search for user defined
% function
catch ME
    line = ME.stack.line;
    if (strcmp(ME.identifier,'') && (line == 134))
        userstatfun         = str2fun(cfg.statistic{1});
        nout                = nargout(userstatfun);
        outputs             = cell(1, nout);
        [outputs{:}]        = userstatfun(cfg,cv);
        stat.statistic      = outputs;
    end
end


% get the model averaged over folds
stat.model  = cv.model;
stat.result = cv.result; %keep this information to be able to inspec per example performance without having to save entire cv object
if exist('out') stat.out    = out; end

% fn = fieldnames(stat.model{1});
% if any(strcmp(fn, 'weights'))
%     % create the 'encoding' matrix from the weights, as per Haufe 2014.
%     covdat = cov(dat');
%     for i=1:length(stat.model)
%         W = stat.model{i}.weights;
%         M = dat'*W;
%         covM = cov(M);
%         stat.model{i}.weightsinv = covdat*W/covM;
%     end
% end
% 
% fn = fieldnames(stat.model{1}); % may now also contain weightsinv
% for i=1:length(stat.model)
%     for k=1:length(fn)
%         if numel(stat.model{i}.(fn{k}))==prod(cfg.dim)
%             stat.model{i}.(fn{k}) = reshape(stat.model{i}.(fn{k}),cfg.dim);
%         end
%     end
% 
% end

% required
stat.trial = [];

% add some stuff to the cfg
cfg.cv = cv;
