tic
clc
figure(1)
clf
clear all
format long

%------------------------        parameters        ------------------------
% before using this function you must specified your function in fun00.m
% file in current directory and then set the parameters
var=3;            % Number of variables (this item must be equal to the
                  %   number of variables that is used in the function in
                  %   fun00.m file)
n=100;            % Number of population

m0=20;            % Number of generations that max value remains constant
                  %   (use for termination criteria)
nmutationG=20;                  %number of mutation children(Gaussian)
nmutationR=20;                  %number of mutation children(random)
nelit=3;                        %number of elitism children
valuemin=[1,0,1];     % min possible value of variables
valuemax=[5,2,2];      % max possible value of variables

%-------------------------------------------------------------------------
nmutation=nmutationG+nmutationR;
sigma=(valuemax-valuemin)/10;    %Parameter that related to Gaussian
                                 %   function and used in mutation step
max1=zeros(nelit,var);
parent=zeros(n,var);
cu=[valuemin(1) valuemax(1) valuemin(2) valuemax(2) valuemin(3) valuemax(3)];
for l=1:var
    p(:,l)=valuemin(l)+rand(n,1).*(valuemax(l)-valuemin(l));
end
initial=p;
m=m0;
maxvalue=ones(m,1)*-1e10;
maxvalue(m)=-1e5;
g=0;
meanvalue(m)=0;