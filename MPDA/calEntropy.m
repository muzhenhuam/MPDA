function res = calEntropy(mix)
[m, n]=size(mix);
h=zeros(m,n);

for i=1:m
    for j=1:n
       h(i,j) =   - mix(i,j).*log(mix(i,j));

    end;
end;
res = abs(sum(sum(h)));