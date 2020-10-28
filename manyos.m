%% CLEANING UP

clear all;close all;clc

%% DEPENDENCIES SPECIFICATION

% DATA PATH SPECIFICATION
ToolsPath                   = '/home/guille/GitHub/SoO/Tools';
DataPath                    = '/media/guille/DADES/DADES/PhysioNet/QTDB/manual0';

% Wavedet codes
addpath([ToolsPath filesep 'BSB' filesep 'WAVEDET' filesep 'WAVEDET_MULTI_LEAD' filesep 'devel']);
addpath([ToolsPath filesep 'BSB' filesep 'WAVEDET' filesep 'WAVEDET_DETECTION_TOOLS' filesep 'ver2']);
% BSB's Input/output codes
addpath([ToolsPath filesep 'BSB' filesep 'tools' filesep 'input_output']);
addpath([ToolsPath filesep 'BSB' filesep 'tools' filesep 'input_output' filesep 'ver3']);
addpath([ToolsPath filesep 'BSB' filesep 'tools' filesep 'ecgtb_new' filesep 'ver2']);
addpath([ToolsPath filesep 'BSB' filesep 'tools' filesep 'others' filesep 'ver1']);

% Load auxiliary functions
addpath(ToolsPath);
addpath([ToolsPath filesep 'ECG']);
addpath([ToolsPath filesep 'LIBSVM']);
addpath([ToolsPath filesep 'WaveletTools']);

%% EXECUTION

% INITIALIZATION PARAMETERS
parameters = update_parameters(struct(),'miliseconds','Fs',1000,'Leads',12,     ...
                               'CroppingWindow',250,'QRStolerance',100,         ...
                               'NumLeads',2,'NumNeighbors',2,'DeltaQ',100,       ...
                               'DeltaS',100,'DeltaT',100,'RefractoryPeriod',200,  ...
                               'Directory',dir([DataPath filesep '*.txt']),     ...
                               'Similarity',0.9);
parameters = update_parameters(parameters,'CovarianceWindow',round(parameters.CroppingWindow/3));


%% Load dataset

% Dataset
dataset = readtable([DataPath filesep 'Dataset.csv']);
dataset = dataset(:,2:end);

% Validity
validity = readtable([DataPath filesep 'ValidityMatlab.csv'],'delimiter',',');

%%

nelem = size(dataset,1);
nfiles = size(dataset,2);

% Get unique names
unames = cell(nfiles,1);
for i = 1:nfiles
    name = dataset.Properties.VariableNames{i};
    id = split(name,'_');
    unames{i} = id{1};
end
unames = unique(unames);

%%

for i = 1:length(unames)
    % Retrieve identifiers
    id = unames{i};
    id_0 = [id, '_0'];
    id_1 = [id, '_1'];
    
    % Locate in dataset
    i_0 = strcmpi(dataset.Properties.VariableNames,id_0);
    i_1 = strcmpi(dataset.Properties.VariableNames,id_1);
    
    % Locate validity
    valid_loc_0 = strcmpi(validity{:,1},id_0);
    valid_loc_1 = strcmpi(validity{:,1},id_1);
    valid_0 = validity{valid_loc_0,2:end};
    valid_1 = validity{valid_loc_1,2:end};
    valid_0 = valid_0(~isnan(valid_0));
    valid_1 = valid_1(~isnan(valid_1));
    
    if ~allclose(valid_0,valid_1)
        error("Check validity file")
    else
        valid = valid_0;
    end
    
    % Compute onsets and offsets
    onsets = valid(1:fix(length(valid)/2));
    offsets = valid(fix(length(valid)/2)+1:end);
    
    % Define output structure
    AnotSLStruct_0 = struct();
    AnotSLStruct_1 = struct();
    
    AnotSLStruct_0.P = [];
    AnotSLStruct_0.Pon = [];
    AnotSLStruct_0.Poff = [];
    AnotSLStruct_0.qrs = [];
    AnotSLStruct_0.QRSon = [];
    AnotSLStruct_0.QRSoff = [];
    AnotSLStruct_0.T = [];
    AnotSLStruct_0.Ton = [];
    AnotSLStruct_0.Toff = [];

    AnotSLStruct_1.P = [];
    AnotSLStruct_1.Pon = [];
    AnotSLStruct_1.Poff = [];
    AnotSLStruct_1.qrs = [];
    AnotSLStruct_1.QRSon = [];
    AnotSLStruct_1.QRSoff = [];
    AnotSLStruct_1.T = [];
    AnotSLStruct_1.Ton = [];
    AnotSLStruct_1.Toff = [];
    
    % Iterate over validity regions
    for j = 1:length(onsets)
        on = onsets(j);
        off = offsets(j);

        Signal = [dataset{on:off,i_0}, dataset{on:off,i_1}];

        % Refine signal
        [B,A]  = butter(4, 125.0/parameters.Fs,'low');
        Signal = filtfilt(B,A,Signal);
        [B,A]  = butter(4, 2.5/parameters.Fs,'high');
        Signal = filtfilt(B,A,Signal);
        
        % Define header
        Header.nsamp      = size(Signal,1);
        Header.nsig       = size(Signal,2);
        Header.channels   = string({'I';'II'});
        Header.freq       = 250.0;
        Header.Decimation = 1;
        
        [AnotSL,AnotML] = QRSDelAndRules(Signal, Header,            ...
                                         parameters.QRStolerance,   ...
                                         parameters.NumLeads,       ...
                                         parameters.NumNeighbors,   ...
                                         parameters.DeltaQ,         ...
                                         parameters.DeltaS,         ...
                                         parameters.DeltaT);
        
        % Recover as structs
        AnotSLStruct_0.P = [AnotSLStruct_0.P; AnotSL{1}.P' + on];
        AnotSLStruct_0.Pon = [AnotSLStruct_0.Pon; AnotSL{1}.Pon' + on];
        AnotSLStruct_0.Poff = [AnotSLStruct_0.Poff; AnotSL{1}.Poff' + on];
        AnotSLStruct_0.qrs = [AnotSLStruct_0.qrs; AnotSL{1}.qrs' + on];
        AnotSLStruct_0.QRSon = [AnotSLStruct_0.QRSon; AnotSL{1}.QRSon' + on];
        AnotSLStruct_0.QRSoff = [AnotSLStruct_0.QRSoff; AnotSL{1}.QRSoff' + on];
        AnotSLStruct_0.T = [AnotSLStruct_0.T; AnotSL{1}.T' + on];
        AnotSLStruct_0.Ton = [AnotSLStruct_0.Ton; AnotSL{1}.Ton' + on];
        AnotSLStruct_0.Toff = [AnotSLStruct_0.Toff; AnotSL{1}.Toff' + on];
        
        AnotSLStruct_1.P = [AnotSLStruct_1.P; AnotSL{2}.P' + on];
        AnotSLStruct_1.Pon = [AnotSLStruct_1.Pon; AnotSL{2}.Pon' + on];
        AnotSLStruct_1.Poff = [AnotSLStruct_1.Poff; AnotSL{2}.Poff' + on];
        AnotSLStruct_1.qrs = [AnotSLStruct_1.qrs; AnotSL{2}.qrs' + on];
        AnotSLStruct_1.QRSon = [AnotSLStruct_1.QRSon; AnotSL{2}.QRSon' + on];
        AnotSLStruct_1.QRSoff = [AnotSLStruct_1.QRSoff; AnotSL{2}.QRSoff' + on];
        AnotSLStruct_1.T = [AnotSLStruct_1.T; AnotSL{2}.T' + on];
        AnotSLStruct_1.Ton = [AnotSLStruct_1.Ton; AnotSL{2}.Ton' + on];
        AnotSLStruct_1.Toff = [AnotSLStruct_1.Toff; AnotSL{2}.Toff' + on];
    end
    
    % Write to file
    AnotSLTable_0 = struct2table(AnotSLStruct_0);
    AnotSLTable_1 = struct2table(AnotSLStruct_1);
    
    writetable(AnotSLTable_0,['MANYOS/',id_0,'.txt'])
    writetable(AnotSLTable_1,['MANYOS/',id_1,'.txt'])
end
