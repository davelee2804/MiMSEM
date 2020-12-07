% TODO: re-arrange the eigenvalues by projecting onto 
%       fourier modes

clear
fid = fopen('quad_op_04.txt','r');
data=fscanf(fid,'%f');
fclose(fid);
C=reshape(data,300,[]);
A=eye(300);
B=0.0*C;
e1=polyeig(C,B,A);
dx=1.0/300.0;
B=(-1.0i*2.0*dx/(2.0*pi))*C;
n=length(e1)
a=sort(abs(real(e1)))/n;
b=a(599)
%B=(-1.0i*b*2.0*dx*2.0*dx/(2.0*pi*2.0*pi))*C;
%v4=0.072*(dx**3.2)
%v2=v4**0.5
%B=(v2*1.0i)*C;
e2=polyeig(C,B,A);
f1=sort(abs(real(e1)))/n;
g1=sort(abs(imag(e1)))/n;
f2=sort(abs(real(e2)))/n;
g2=sort(abs(imag(e2)))/n;
k=linspace(0,1,600);
plot(k,f1,k,f2,k,g2,k,0.5*pi*k)
