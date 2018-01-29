
%%% no. TRs:  231
% test hellooooo

%%%% THis version generates the noise patterns during the experiment

clc
clear all
close all

%%% PREPARE AND COLLECT INFO
echo off
clear all
KbName('UnifyKeyNames')
Screen('Preference', 'SkipSyncTests', 1);
input('hit enter to begin...  ');

%%%%%%%%

output.Subject = 'tmp';
saveFolder = sprintf('./DataSbj%s',output.Subject);

%%%%%%%%

eyeTrackON = 0; % 1 == start eye tracking, 0 = no eyetracking
realScan = 0; % 0 = not a scan, 1 = real scan

%%%%%%%%
[keyboardIndices, productNames, ~] = GetKeyboardIndices;
if realScan == 1, deviceString=  'Xkeys';
else deviceString = 'Apple Internal Keyboard / Trackpad';end            % imac: 'Apple Wireless Keyboard'  , laptop: 'Apple Internal Keyboard / Trackpad'

for i=1:length(productNames)    %for each possible device
    if strcmp(productNames{i},deviceString)     %compare the name to the name you want
        deviceNumber = keyboardIndices(i);        %grab the correct id, and exit loop
        break;
    end
end

deviceNumber = 2;
if deviceNumber==0 %%error checking
    error('No device by that name was detected');
end

triggerKey = 46;                % KbName('=+');
keyList = ones(1,256);          % keys for KbQueueCreate
keyList(triggerKey) = 0;        % dont want to record trigger

if realScan == 1, keyPressNumbers = {'30', '31'}; % index: counter, middle: clock
elseif realScan == 0, keyPressNumbers = {'80', '79'}; end %{'80', '79'} for arrows on macbook.   key{1} (left arrow) for counter-clockwise and key{2}(right arrow) for clockwise

%%%%%%%%%
%%% SCREEN PARAMETERS
w.whichScreen = 0;

if realScan == 1,
    % parameters for BU scanner:
    w.ScreenWidth = 51.2;         % horizontal display size
    w.ViewDistance = 82;        % in cm, ideal distance: 1 cm equals 1 visual degree (at 57 cm)
else
    w.ScreenWidth = 33;         % horizontal display size - 41.5 in scanner;
    w.ViewDistance = 57;        % in cm, ideal distance: 1 cm equals 1 visual degree (at 57 cm) - 107.5 at scanner with eye-tracking, 98 normal screen
end

w.frameRate = 60;
w.ScreenSizePixels = Screen('Rect', w.whichScreen); %Scanner display = [0 0 1024 768];
w.VisAngle = (2*atan2(w.ScreenWidth/2, w.ViewDistance))*(180/pi); % Visual angle of the whole screen
stim.ppd = round(w.ScreenSizePixels(3)/w.VisAngle); % pixels per degree visual angle

%%%%%%%%
%%% STIMULUS PARAMETERS
stim.sizeDeg = 15;         % in visual degree
stim.annulusDeg = 3;
stim.fixationDeg = 0.7;
stim.sizePix = round(stim.sizeDeg*stim.ppd);     % in pixels
stim.annulusPix = round(stim.annulusDeg*stim.ppd);
stim.fixationPix = round(stim.fixationDeg*stim.ppd);
stim.outerFixationPix = round(1.4 * stim.fixationPix);
stim.fixationDotPix = round(.06*stim.ppd); % in pixel
stim.grey = 128;
stim.contrast = .1;
stim.amp = stim.contrast * stim.grey;
stim.freqCPD = 2;
stim.tilt = 45;     % absolute stimulus orientation
stim.devTilt = 16;  % magnitude of change in the angle of the grating (in degrees)
stim.numOrient = 2;     % 45 and -45

%%%%% noise characteristics:
noise.maxContrast = (.99-stim.contrast);
noise.minContrast = 0;
noise.numLevels = 2;
noise.contrastLevels = [noise.minContrast,noise.maxContrast]; %logspace(log10(noise.minContrast),log10(noise.maxContrast),noise.numLevels-1)];

%%%% RSVP task:
RSVP.distractorLetters = ['X' 'L' 'V' 'H' 'S' 'A' 'C' 'P' 'Z' 'Y'];
RSVP.targetLetter = ['J', 'K'];
RSVP.probTarget = 0.3;
% size 1 letter point in visual angle
% (fontsize is in "point" (not pixel) and "1 point = 1/72th of 1 inch" and 1inch=2.54cm)
pointSize = (2*atan2((2.54/72)/2, w.ViewDistance))*(180/pi);
stim.letterSizeDegree = .5;
stim.letterSizePoint = round(stim.letterSizeDegree/pointSize);   % in letter points

%%%%%%%%%
%%% TIMING PARAMETERS
t.theDate = datestr(now,'yymmdd');  %Collect todays date
t.timeStamp = datestr(now,'HHMM');  %Timestamp for saving out a uniquely named datafile (so you will never accidentally overwrite stuff)

t.MySeed = sum(100*clock);
rng('default'); 
rng(t.MySeed);

t.TR = 2;                           % TR length
t.blockDur = 16;
t.rsvpDurPerLetter = .2;               % duration of one letter presentatoin during RSVP
t.responseDur = 1.5;                % response time
t.initBlank = 30;           % initial fixation period before starting showing the stimulus
t.changeDur = .5;           % duration of the change in orientation
t.flickeringFreq = 10;          % Hz
t.infoBlock = 2;      % information block: subject is told which task should be done (orientation or RSVP)
RefreshDur = 1/w.frameRate;
t.interChangeInt = 2;        % minimum time between two consecutive change in tilt
t.confInterval  = 1.5;    % no change during this period at the beginning and double of this period at the end of the block
numAttCondition = 2;        % number of attention condition
t.numRepPerNoiseLevel = 3;
t.numBlocks = noise.numLevels*t.numRepPerNoiseLevel*stim.numOrient*numAttCondition;

numInitFrames = t.initBlank/RefreshDur;
numStimFrames = t.blockDur/RefreshDur;
numIBFrames = t.infoBlock/RefreshDur;

whenStartStim = (t.initBlank + t.infoBlock) : (t.blockDur + t.infoBlock) : (t.numBlocks-1)*(t.blockDur + t.infoBlock) + (t.initBlank + t.infoBlock);
whenStopStim = whenStartStim + t.blockDur;
whenStartIBI = t.initBlank:(t.infoBlock + t.blockDur) : (t.numBlocks-1)*(t.blockDur + t.infoBlock) + t.initBlank;
whenStopIBI = whenStartIBI + t.infoBlock;

t.totalTime = t.numBlocks*(t.blockDur + t.infoBlock) + t.initBlank;
output.TotalTRs = t.totalTime/t.TR;

%%%%%%%%
%%% INITIATE & GENERATE DATA FILE
if ~exist(saveFolder,'dir')
    mkdir(saveFolder);
end

checkName = sprintf('%s/*_%s.mat',saveFolder,output.Subject);
runNo = length(dir(checkName)) + 1;
output.fileName = sprintf('Data_NoiseEquivalent_Run%i_%s',runNo,output.Subject);
eyeTrackingFinleName = sprintf('%sNEQ%i',output.Subject,runNo);

%%%%%%%
%%% CREATE Frequency Patterns
% Gaussian Mask
xysize = round(stim.sizePix);
[x,y] = meshgrid((-xysize/2):(xysize/2) - 1, (-xysize/2):(xysize/2) - 1);
gaussian_std = round(stim.ppd*2);
eccen = sqrt((x).^2 + (y).^2); 	% calculate eccentricity of each point in grid relative to center of 2D image
Gaussian = zeros(xysize); Gaussian(eccen <= (xysize/2) - stim.ppd/2) = 1;
Gaussian = conv2(Gaussian, fspecial('gaussian', stim.ppd, stim.ppd), 'same');

% Gaussian Annulus
[X,Y] = meshgrid(1:xysize,1:xysize);
Annulus = zeros(xysize);
r_eccen = sqrt((X - xysize/2).^2 + (Y - xysize/2).^2); 	% calculate eccentricity of each point in grid relative to center
Annulus(r_eccen > stim.annulusPix/2) = 1;
Annulus = conv2(Annulus, fspecial('gaussian', 30, 10), 'same');
Annulus(r_eccen > stim.annulusPix) = 1;

% Setup Filters
Mask = Gaussian.*Annulus;
Mask(Mask<0.1) = 0;
freqSample = round(stim.freqCPD.*stim.sizeDeg)/xysize;     % cycles/sample

%%%%%%
% parameters of events:
events.letterPresOn = 1;        % whether the letters are shown in the center of the stimulus or it will be just blank
events.maxNumChangePerBlock = 4;       % maximum number of changes in tilt in each block

%%%%
numChangeInBlock = randi(events.maxNumChangePerBlock,1,t.numBlocks);       % no. of changes in each block
[tChangeSet tiltDirSet] = GenerateChangeTimes(t.blockDur,t.confInterval,t.interChangeInt,numChangeInBlock);
output.tChangeSet = tChangeSet;
output.tiltSet = tiltDirSet;
%tiltDirection = Shuffle(repmat([1;-1], ceil(sum(numChangeInBlock)/2), 1));    % whether the change is clockwise or counter-clockwise
%%%

conditionMat = zeros(t.numBlocks,3);       % columns--> 1: attended or unattended, 2:stim orientation, 3: noise level
numBlocksWithoutRep = t.numBlocks/t.numRepPerNoiseLevel;

attendedOn = [ones(numBlocksWithoutRep/numAttCondition,1);zeros(numBlocksWithoutRep/numAttCondition,1)];
stimOrientation = repmat([stim.tilt.*ones(length(attendedOn)/4,1);-stim.tilt.*ones(length(attendedOn)/4,1)],numAttCondition,1);
noiseLevel = repmat((1:noise.numLevels)',2*numAttCondition,1);

attendedOnWithRep  = repmat(attendedOn,t.numRepPerNoiseLevel,1);
stimOrientationWithRep = repmat(stimOrientation,t.numRepPerNoiseLevel,1);
noiseLevelWithRep = repmat(noiseLevel,t.numRepPerNoiseLevel,1);

conditionMat(:,1) = attendedOnWithRep;
conditionMat(:,2) = stimOrientationWithRep;
conditionMat(:,3) = noiseLevelWithRep;

conditionMat = conditionMat(randperm(t.numBlocks),:);
output.conditionMat = conditionMat;

%%% stimulus:

sineStim = cos(2.*pi.*freqSample.*cosd(0).*X + 2.*pi.*freqSample.*sind(0).*Y);

%%% WINDOW SETUP
AssertOpenGL;       % to verify that this psychtoolbox works based on OpenGL (some versions don't)
[window, rect] = Screen('OpenWindow',w.whichScreen, stim.grey);
HideCursor;

%%% MAKE COLOR LOOKUP TABLE AND APPLY GAMMA CORRECTION
OriginalCLUT = Screen('LoadCLUT', window);
green = [0 255 0]; red = [255 0 0];
white = WhiteIndex(w.whichScreen);
black = BlackIndex(w.whichScreen);

if realScan == 1
    MyCLUT = load('/Users/linglab/Documents/Experiments/BU_imagingCenter/CRF_tunedNormalization/linearizedCLUT.mat');
    Screen('LoadNormalizedGammaTable', window, abs(MyCLUT.linearizedCLUT));
else
    MyCLUT = load('linearizedCLUT.mat');        % !!!! make sure to update this
    Screen('LoadNormalizedGammaTable', window, MyCLUT.linearizedCLUT);
end

%%%%%%%%%%%%%%%%%%%
%%% If eyeTrackOn == 1, eye link setup
if eyeTrackON == 1
    [el, edf_filename] = eyeTrackingOn(window, eyeTrackingFinleName, rect, stim.ppd);
end


%%% CREATE PATCHES
centerX = w.ScreenSizePixels(3)/2;
centerY = w.ScreenSizePixels(4)/2;
Screen('TextStyle', window, 1);
Screen('TextSize', window, stim.letterSizePoint);
bbox = Screen('TextBounds', window, 'X');               % gives the size of the bounding box of the text specified
newRect = CenterRectOnPoint(bbox, centerX, centerY);    % offset the rect to center it on the x and y stated positions
tx = newRect(1); ty = newRect(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% START THE EXPERIMENT%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Initialize run
rsvpCounter = 0;
odCounter = 0;
rsvpTime = 0;
whatLetter = RandSample(RSVP.distractorLetters);
stimSign = 1;
tiltDeviationPhaseOld = 0;

% parameters for testing the code:
log.nChanges = zeros(1,t.numBlocks);
log.tStChange = nan(t.numBlocks,events.maxNumChangePerBlock);
log.tEnChange = nan(t.numBlocks,events.maxNumChangePerBlock);
nTotalChange = 0;

% wait for backtick sync with scanner
Screen('FillOval', window, white, CenterRectOnPoint([0 0 stim.outerFixationPix stim.outerFixationPix], centerX, centerY));
Screen('FillOval', window, stim.grey, CenterRectOnPoint([0 0 stim.fixationPix stim.fixationPix], centerX, centerY));
Screen('DrawText', window, '~', tx, ty + 40, 255, 128);
Screen('Flip', window);

KbTriggerWait(triggerKey, deviceNumber);

PsychHID('KbQueueCreate', deviceNumber, keyList);

%%%%%%%%%%%%%%%%%%%
%%% If eyeTrackOn == 1, start recording
if eyeTrackON == 1
    [status, el] = eyeTrackingRecord(el, rect, stim.ppd);
end

%%%%%%%%%%%%%%%%%%%
%%% INTIAL BASELINE----------------------------------------------------
startTime = GetSecs;
log.startTime = startTime;
tic;
for ii=1:numInitFrames
    
    Screen('FillOval', window, white, CenterRectOnPoint([0 0 stim.outerFixationPix stim.outerFixationPix], centerX, centerY));
    Screen('FillOval', window, stim.grey, CenterRectOnPoint([0 0 stim.fixationPix stim.fixationPix], centerX, centerY));
    Screen('FillOval', window, green, CenterRectOnPoint([0 0 stim.fixationDotPix stim.fixationDotPix], centerX, centerY));
    if (GetSecs > startTime + t.initBlank)
        break;
    end
    Screen('Flip', window, 0, [], 1);
    
end
foo = toc;
log.durInitBlank = foo;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Block representation starts:

for blockCounter = 1:t.numBlocks
    
    allowTarget = 0;
    responsewindowTrigODTask = 0;
    responsewindowTrigRSVP = 0;
    nFlick = 1;
    
    attBlock = conditionMat(blockCounter,1);
    baseAngle = conditionMat(blockCounter,2);
    whichNoiseLevel = conditionMat(blockCounter,3);
    
    
    changeTimes = tChangeSet{blockCounter};
    tiltDirection = tiltDirSet{blockCounter};
    nChange = 1;
    
    tic;
    for kk=1:numIBFrames
        
        if GetSecs > startTime+whenStopIBI(blockCounter)
            break
        end
        
        Screen('FillOval', window, white, CenterRectOnPoint([0 0 stim.outerFixationPix stim.outerFixationPix], centerX, centerY));
        Screen('FillOval', window, stim.grey, CenterRectOnPoint([0 0 stim.fixationPix stim.fixationPix], centerX, centerY));
        Screen('FillOval', window, green, CenterRectOnPoint([0 0 stim.fixationDotPix stim.fixationDotPix], centerX, centerY));
        
        if attBlock==0
            whatLetter = 'R';
            rsvpOn = 1;
            odTaskOn = 1 - rsvpOn;
        else
            whatLetter = 'O';
            rsvpOn = 0;
            odTaskOn = 1 - rsvpOn;
        end
        Screen('DrawText', window, whatLetter, tx, ty, 0, 128);
        Screen('Flip', window, 0, [], 1);
        
    end
    foo = toc;
    
    log.durInfoBlock(blockCounter) = foo;
    
    clear flickTimeVec;
    flickTimeVec = GetSecs:1/(2*t.flickeringFreq):(GetSecs + t.blockDur + .1);      % flicking times for the whole coming block
    
    log.tStStimBlock(blockCounter) = GetSecs - startTime;
    tic;
    for kk = 1:numStimFrames
        
        flickTime = flickTimeVec(nFlick);
        if GetSecs > startTime + whenStopStim(blockCounter)
            break
        end
        
        if GetSecs >= flickTime
            stimSign = -1*stimSign;
            nFlick = nFlick + 1;
        end
        
        % stimulus generation:
        noiseStim = (2.*noise.contrastLevels(whichNoiseLevel).*rand(xysize, xysize) - noise.contrastLevels(whichNoiseLevel)).*stim.grey;
        texPointer = Screen('MakeTexture', window,stimSign.*(noiseStim + sineStim*stim.amp) .* Mask + stim.grey);
        
        %%%%%%%% finding the angle of the stimulus. Is it the change period
        %%%%%%%% or not? Should it deviate or not...
        
        tStChange = startTime + whenStartStim(blockCounter) + changeTimes(nChange);
        tEnChange = tStChange + t.changeDur;
        
        if GetSecs > tStChange && GetSecs < tEnChange
            rotAng = baseAngle + tiltDirection(nChange)*stim.devTilt;
            tiltDeviationPhase = 1;
        else
            rotAng = baseAngle;
            tiltDeviationPhase = 0;
        end
        
        if (tiltDeviationPhase == tiltDeviationPhaseOld + 1)
            log.tStChange(blockCounter,nChange) = GetSecs - startTime;
            if odTaskOn
                odCounter = odCounter + 1;
                responsewindowTrigODTask = 1;
                PsychHID('KbQueueStart', deviceNumber);
                responsewindowODTask = GetSecs + t.changeDur + t.responseDur;
            end
        elseif (tiltDeviationPhase == tiltDeviationPhaseOld - 1)
            log.nChanges(blockCounter) = log.nChanges(blockCounter) + 1;
            log.tEnChange(blockCounter,nChange) = GetSecs - startTime;
            nChangeOld = nChange;
            nTotalChange = nTotalChange + 1;
            if nChange< length(tChangeSet{blockCounter})
                nChange = nChange + 1;
            end
        end
        tiltDeviationPhaseOld = tiltDeviationPhase;
        
        %%%%% Presenting letters in the center of the fixation:
        
        if GetSecs < startTime+whenStartStim(blockCounter)+1 || GetSecs > startTime+whenStopStim(blockCounter)-2
            allowTarget = 0;
        end
        
        
        if GetSecs > rsvpTime
            allowTarget = allowTarget + 1;
            if allowTarget > 10
                rsvpTargetOrNot = rand < RSVP.probTarget;
            else
                rsvpTargetOrNot = 0;
            end
            
            if rsvpTargetOrNot
                letterInd = RandSample([1 2]);
                whatLetter = RSVP.targetLetter(letterInd);
                allowTarget = 0;
                if rsvpOn == 1
                    PsychHID('KbQueueStart', deviceNumber);
                    responsewindowTrigRSVP = 1;
                    responsewindowRSVP = GetSecs + t.responseDur;
                    rsvpCounter = rsvpCounter + 1;
                    log.whatletter(rsvpCounter) = whatLetter;
                end
            end
            if ~rsvpTargetOrNot
                whatLetterTmp = RandSample(RSVP.distractorLetters);
                while strcmp(whatLetter, whatLetterTmp)
                    whatLetterTmp = RandSample(RSVP.distractorLetters);
                end
                whatLetter = whatLetterTmp;
            end
            rsvpTime = GetSecs + t.rsvpDurPerLetter;
        end
        
        Screen('DrawTexture', window,texPointer, [], [], rotAng);
        Screen('FillOval', window, white, CenterRectOnPoint([0 0 stim.outerFixationPix stim.outerFixationPix], centerX, centerY));
        Screen('FillOval', window, stim.grey, CenterRectOnPoint([0 0 stim.fixationPix stim.fixationPix], centerX, centerY));
        Screen('DrawText', window, whatLetter, tx, ty, 255, 128);         %xpos,ypos,color,backgroundColor
        Screen('FillOval', window, green, CenterRectOnPoint([0 0 stim.fixationDotPix stim.fixationDotPix], centerX, centerY));
        Screen('Flip', window, 0, [], 1);
        
        %
        if responsewindowTrigRSVP == 1;
            
            if GetSecs >= responsewindowRSVP
                
                [pressed, firstpress] = PsychHID('KbQueueCheck', deviceNumber);
                whichkeys = find(firstpress);
                
                if length(whichkeys)>1      % if two keys were pressed almost simulatneously check for the timing and take the one pressed slightly faster
                    if firstpress(whichkeys(1))<=firstpress(whichkeys(2))
                        whichkeys = whichkeys(1);
                    else
                        whichkeys = whichkeys(1);
                    end
                end
                
                if ~isempty(whichkeys)
                    
                    output.keyPressedRSVP(rsvpCounter) = whichkeys(1);
                    output.timePressedRSVP(rsvpCounter) = firstpress(whichkeys(1)) - startTime;       % time of pressing the key
                    
                    if ((strcmp(num2str(whichkeys), keyPressNumbers{1}) && letterInd==1) || (strcmp(num2str(whichkeys), keyPressNumbers{2}) && letterInd==2))
                        output.rsvpCorrect(rsvpCounter) = 1;
                    else
                        output.rsvpCorrect(rsvpCounter) = 0;
                    end
                else
                    output.rsvpCorrect(rsvpCounter) = nan;
                end
                
                KbQueueFlush();
                responsewindowTrigRSVP = 0;
            end
            
        end
        
        if responsewindowTrigODTask == 1;
            
            if GetSecs >= responsewindowODTask
                
                [pressed, firstpress] = PsychHID('KbQueueCheck', deviceNumber);
                whichkeys = find(firstpress);
                
                if length(whichkeys)>1
                    if firstpress(whichkeys(1))<=firstpress(whichkeys(2))
                        whichkeys = whichkeys(1);
                    else
                        whichkeys = whichkeys(1);
                    end
                end
                
                if ~isempty(whichkeys)
                    
                    output.keyPressedOD(odCounter) = whichkeys;
                    output.timePressedOD(odCounter) = firstpress(whichkeys) - startTime;       % time of pressing the key
                    
                    if ((strcmp(num2str(whichkeys), keyPressNumbers{1}) && tiltDirection(nChangeOld)==-1) || (strcmp(num2str(whichkeys), keyPressNumbers{2}) && tiltDirection(nChangeOld)==1))
                        output.odCorrect(odCounter) = 1;
                    else
                        output.odCorrect(odCounter) = 0;
                    end
                else
                    output.odCorrect(odCounter) = nan;
                end
                
                KbQueueFlush();
                responsewindowTrigODTask = 0;
            end
            
        end
        Screen('Close', texPointer);
    end
    foo = toc;
    log.durStim(blockCounter) = foo;
end

log.totalScanDuration = GetSecs - startTime;

TheData.stim = stim;
TheData.t = t;
TheData.w = w;
TheData.log = log;
TheData.noise = noise;
TheData.RSVP = RSVP;
TheData.output = output;

save([saveFolder,'/',output.fileName]);

Screen('FillOval', window, white, CenterRectOnPoint([0 0 stim.outerFixationPix stim.outerFixationPix], centerX, centerY));
Screen('FillOval', window, stim.grey, CenterRectOnPoint([0 0 stim.fixationPix stim.fixationPix], centerX, centerY));
Screen('TextSize', window,16);
Screen('DrawText', window, 'DONE.', centerX-20, centerY-stim.annulusPix/2,255,128);
Screen('Flip', window, 0, [], 1);

WaitSecs(1);

%%% close screen
ShowCursor
Screen('CloseAll')


%%%%%%%%%%%%%%%%%%%
if eyeTrackON == 1
    Eyelink('StopRecording');
    Eyelink('CloseFile');
    Eyelink('ReceiveFile',edf_filename);
end
%%%%%%%%%%%%%%%%%%%

disp('------------------------------------------------');
disp(['OD Task Accuracy:' num2str(nanmean(output.odCorrect))]);
disp(['Number of Missed Ones: (' num2str(sum(isnan(output.odCorrect))),'/', num2str(numel(output.odCorrect)), ')']);
disp('------------------------------------------------');

disp('------------------------------------------------');
disp(['RSVP Task Accuracy:' num2str(nanmean(output.rsvpCorrect))]);
disp(['Number of Missed Ones: (' num2str(sum(isnan(output.rsvpCorrect))),'/', num2str(numel(output.rsvpCorrect)), ')']);
disp('------------------------------------------------');


%{
% to compute the total number of changes in orientation happened during the attended blocks:
nChangeAttBlocks=0;
for ii=1:t.numBlocks
    if conditionMat(ii,1)==1
        nChangeAttBlocks = nChangeAttBlocks + length(tChangeSet{ii});
    end
end
%}

