%%%% This function gives some time points within a period %%%%
%%% there should be a minimum refractoryTime between two time points %%%%
%%% the way the function works: divides the period into a set of time periods with the same duration. The first
%%% point is selected within the first period then the next period in which
%%% the second time point will be found is updated based on the first point-->
%%% second period: (time of the first change + refracotry time) as the
%%% lower limit and the upper limit of the second time interval (derived
%%% from division) as the upper limit.

function [timeChangeSetBlock dirSetBlock] = GenerateChangeTimes(blockDur,confInterval,refractoryTime,numChangeInBlocks)

probClockwise = .5;

minBeginChangeTime = confInterval;
maxBeginChangeTime = blockDur - 2.*confInterval;
numBlocks = length(numChangeInBlocks);

changeInterval = maxBeginChangeTime - minBeginChangeTime;

for ii=1:numBlocks  
    changeDurLimit = (changeInterval/numChangeInBlocks(ii));
    maxTimes = (minBeginChangeTime + changeDurLimit):changeDurLimit:maxBeginChangeTime;
    minTime = minBeginChangeTime;
    
    for jj=1:numChangeInBlocks(ii)
        timeChangeInBlock(jj) = minTime + (maxTimes(jj) - minTime)*rand;
        minTime = timeChangeInBlock(jj) + refractoryTime;     % next change won't happen until the refractory period finishes
    
    end
    timeChangeSetBlock{ii} = timeChangeInBlock;
    dirSetBlock{ii} = (rand(1,numChangeInBlocks(ii))<probClockwise).*2 - 1;
    clear timeChangeInBlock maxTimes
end