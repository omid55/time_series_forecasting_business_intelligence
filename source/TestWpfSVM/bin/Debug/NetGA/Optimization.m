function [ output ] = Optimization( arg,trainInputs,trainTargets,testInputs,testTargets )

numLayers=round(arg(1,1));
numNeurons=round(arg(1,2));
layers=ones(1,numLayers)*numNeurons;  
net=newff(trainInputs,trainTargets,layers,{},'trainlm');
net.trainParam.lr=0.05;
net.trainParam.mc=0.9;
net.trainParam.show=25;
net.trainParam.epochs=1000;
net.trainParam.goal=0;
net.trainParam.max_fail=6;
net.trainParam.min_grad=1e-010;
net.trainParam.time=Inf;
net.trainParam.mu=0.001;
net.trainParam.mu_dec=0.1;
net.trainParam.mu_inc=10;
net.trainParam.mu_max=10000000000;
net.trainParam.mem_reduc=1;
net.trainParam.showCommandLine=false;
net.trainParam.showWindow=false;
net=train(net,trainInputs,trainTargets);
res=sim(net,testInputs);
errors=(res-testTargets);
len=length(errors);
mse=0;
for k=1:len
    mse=mse+errors(k)^2;
end
mse=mse/len;
output=-1*mse;

end

