clear;
folder = "\path_to_FSDD_US_audio\";

%all stationary bg noise
%babble
%birds chirping or crickets
%click noise by equipment
%noiseless
%something different than phoneme
%transient-some noise before or after phoneme

%nothing said

fs = 44100;

aFE = audioFeatureExtractor( ...
    SampleRate=fs, ...,
    Window=hamming(round(0.03*fs),"periodic"), ...
    OverlapLength=round(0.02*fs), ...
    mfcc=true, ...
    mfccDelta=true, ...
    mfccDeltaDelta=true, ...
    gtcc=true,...
    gtccDelta=true,...
    gtccDeltaDelta=true,...
    pitch=true, ...
    zerocrossrate=true, ...
    shortTimeEnergy=true);

%filePattern = fullfile(folder, '*.wav');

filePattern = fullfile(folder, '*.wav');
Files = dir(filePattern);
data = cell(length(Files),1);

for k = 1 : length(Files)
    basefile = Files(k).name;
    File = fullfile(Files(k).folder, basefile);
    [audioIn,fs]= audioread(File);
    [m, n] = size(audioIn);
    if n == 2
        y = sum(audioIn,2);
        peakAmp = max(abs(y)); 
        y = y/peakAmp;
        %  check the L/R channels for orig. peak Amplitudes
        peakL = max(abs(audioIn(:, 1)));
        peakR = max(abs(audioIn(:, 2))); 
        maxPeak = max([peakL peakR]);
        %apply x's original peak amplitude to the normalized mono mixdown 
        y = y*maxPeak;
    
    else
        y = audioIn;
    end

    features = mean(extract(aFE,y),1);
    data{k} = features;
end

data = data(~cellfun('isempty',data));
feature_matrix = cell2mat(data);

class = 2 * ones(height(feature_matrix),1);

feature_matrix = [feature_matrix class];
    
%writematrix(feature_matrix, "D:\Phd\data\New data\Classes\ha\gtcc_zcr_spectral_ha_v.csv", 'WriteMode', 'append')

writematrix(feature_matrix, ".\CSVs_for_testing\filename.csv")

