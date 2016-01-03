s=sort(y);
    maxvalue1(1:nelit)=s(n:-1:n-nelit+1);
    %for t=1:nelit
    %    maxvalue1(t)=s(t);
    %end
    
    if nelit==0
        maxvalue1(1)=s(n);
        for i=1:n
            if y(i)==maxvalue1(1)
                max1(1,:)=p(i,:);
            end
        end
    end
    for k=1:nelit
        for i=1:n
            if y(i)==maxvalue1(k)
                max1(k,:)=p(i,:);
            end
        end
    end
    if var==2
        
        figure(1)
        subplot(2,2,1)
        hold off
        ezmesh('x*sin(abs(x))+y*sin(abs(y))',cu)
        hold on
        plot3(p(:,1),p(:,2),y,'ro')
        plot3(max1(1,1),max1(1,2),maxvalue1(1),'bh')
        title({' Genetic Algorithm '...
            ,'Performance of GA ( o : each individual)'},'color','b')

    end
    
    NowA = max1(1,1)
    NowB = max1(1,2)
    bestfitness = maxvalue1(1)   %it is error
	disp "-----------------------------------------------------"
    
    y=y-min(y)*1.02;
    sumd=y./sum(y);
    meanvalue=y./(sum(y)/n);


    %-------------   ****   Selection: Rolette wheel   ****-------------
    for l=1:n
        sel=rand;
        sumds=0;
        j=1;
        while sumds<sel
            sumds=sumds+sumd(j);
            j=j+1;
        end
        parent(l,:)=p(j-1,:);
    end
    p=zeros(n,var);

    %-------------   ****    regeneration   ****-------------
    for l=1:var


        %-------------   ****    cross-over   ****-------------
        for j=1:ceil((n-nmutation-nelit)/2)
            t=rand*1.5-0.25;
            p(2*j-1,l)=t*parent(2*j-1,l)+(1-t)*parent(2*j,l);
            p(2*j,l)=t*parent(2*j,l)+(1-t)*parent(2*j-1,l);
        end


        %-------------   ****    elitism   ****-------------
        for k=1:nelit
            p((n-nmutation-k+1),l)=max1(k,l);
        end


        %-------------   ****    mutation   ****-------------
        for i=n-nmutation+1:n-nmutationR
            phi=1-2*rand;
            z=erfinv(phi)*(2^0.5);
            p(i,l)=z*sigma(l)+parent(i,l);

        end
        for i=n-nmutationR+1:n
            p(i,1:var)=valuemin(1:var)+rand(1,var).*(valuemax(1:var)-...
                valuemin(1:var));
        end
        for i=1:n
            for l=1:var
                if p(i,l)<valuemin(l)
                    p(i,l)=valuemin(l);
                elseif p(i,l)>valuemax(l)
                    p(i,l)=valuemax(l);
                end
            end
        end
    end
    p;
    m=m+1;
    max1;
    maxvalue(m)=maxvalue1(1);
    maxvalue00(m-m0)=maxvalue1(1);
    mean00(m-m0)=sum(s)/n;
    meanvalue(m)=mean00(m-m0);
    figure(1)
    if var==2
        subplot(2,2,2)
    end
    hold off
    plot(maxvalue00,'b')
    hold on
    plot(mean00,'r')
    hold on
    title({'Performance of GA',...
        'best value GA:blue, best value RS:black, mean value GA:red',''}...
        ,'color','b')
    xlabel('number of generations')
    ylabel('value')