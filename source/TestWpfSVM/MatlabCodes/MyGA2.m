    sigma=sigma./(1.05);% reducing the sigma value
    %  ------  **** % reducing the number of mutation()random   **** ----
    g=g+1;
    if g>10 && nmutationR>0
        g=0;
        nmutationR=nmutationR-1;
        nmutation=nmutationG+nmutationR;
    end
 %-------------   ****    function evaluation   ****-------------