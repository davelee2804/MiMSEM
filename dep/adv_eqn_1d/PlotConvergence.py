#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# N=3, dt = 2.5/(20*n_els)
n_els=np.array([4,8,16,32,64])
dx = 1.0/n_els
a_orig=np.array([0.0033614307716222243,0.0003845292647497689,4.618758026157271e-05,5.708446643156211e-06,7.114713478420709e-07])
a_upir=np.array([0.11900588432388003,0.06083250831200746,0.030618625792267463,0.015334739107917382,0.007670557672606986])
a_uppg=np.array([0.0029901321869783095,0.00029601726670522066,3.554366302641541e-05,4.4006410244777355e-06,5.487794348938929e-07])

fig1,ax1 = plt.subplots()
plt.loglog(dx,a_orig,'-o',c='g')
#plt.loglog(n_els,a_upir,'-o',c='r')
plt.loglog(dx,a_uppg,'-+',c='b')
#plt.loglog(n_els,dx,'-.k')
plt.loglog(dx,dx*dx*dx,'--k')
#plt.legend(['A','A (IR)','A (PG)','$dx^1$','$dx^3$'])
ax1.set_xticks([0.02,0.04,0.06,0.08,0.1,0.2])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.legend(['$A$','$A_{PG;\Delta t}$','$dx^3$'])
plt.title('$L^2$ error convergence, $p=3$')
plt.xlabel('element width, $\Delta x$')
plt.savefig('l2_convergence_p3_revised.pdf')
plt.show()

# N=4, dt = 2.5/(20*n_els)
n_els=np.array([4,8,16,32,64])
dx = 1.0/n_els
a_orig=np.array([0.0003406983660765121,2.347465891428806e-05,1.508015311761844e-06,9.491179062355806e-08,5.942401143949632e-09])
a_upir=np.array([0.11899022249191074,0.0608926559629707,0.030626050331139392,0.015335676432844577,0.007670675514269176])
a_uppg=np.array([0.0002782691863297902,2.1255278041690046e-05,1.4618177448818919e-06,9.412575206816705e-08,5.92983368225117e-09])

plt.loglog(dx,a_orig,'-o',c='g')
#plt.loglog(n_els,a_upir,'-o',c='r')
plt.loglog(dx,a_uppg,'-+',c='b')
#plt.loglog(n_els,dx,'-.k')
plt.loglog(dx,dx*dx*dx*dx,'--k')
#plt.legend(['A','A (IR)','A (PG)','$dx^1$','$dx^4$'])
plt.legend(['$A$','$A_{PG;\Delta t}$','$dx^4$'])
plt.title('$L^2$ error convergence, $p=4$')
plt.xlabel('element width, $\Delta x$')
plt.savefig('l2_convergence_p4.png')
plt.show()

# N=5, dt = 2.5/(20*n_els)
n_els=np.array([4,8,16,32,64])
dx = 1.0/n_els
a_orig=np.array([2.7311323476242296e-05,8.179520641434409e-07,2.5075847136055104e-08,7.794890878995601e-10,2.432606346123907e-11])
a_upir=np.array([0.11899806291187187,0.06089245031523031,0.03062604592726237,0.015335676281573611,0.00767067550951868])
a_uppg=np.array([2.1662246624016687e-05,4.925385233479937e-07,1.462140265593335e-08,4.512138802415079e-10,1.4055728485717257e-11])

plt.loglog(dx,a_orig,'-o',c='g')
#plt.loglog(n_els,a_upir,'-o',c='r')
plt.loglog(dx,a_uppg,'-+',c='b')
#plt.loglog(n_els,dx,'-.k')
plt.loglog(dx,dx*dx*dx*dx*dx,'--k')
#plt.legend(['A','A (IR)','A (PG)','$dx^1$','$dx^5$'])
plt.legend(['$A$','$A_{PG;\Delta t}$','$dx^5$'])
plt.title('$L^2$ error convergence, $p=5$')
plt.xlabel('element width, $\Delta x$')
plt.savefig('l2_convergence_p5.png')
plt.show()

# N=6, dt = 2.5/(40*n_els)
n_els=np.array([4,8,16,32,64])
dx = 1.0/n_els
#a_orig=np.array([1.8132399004108258e-06,3.0194343342832945e-08,4.808481613954586e-10,7.550027186797983e-12,1.18179823506625e-13])
#a_upir=np.array([0.11899604100583266,0.06089240034217338,0.030626041313369025,0.015335676142736364,0.0076706755051379845])
#a_uppg=np.array([1.432900135642253e-06,2.695043570732134e-08,4.643669968504013e-10,7.549935723375382e-12,1.5837515403494719e-12])
a_orig=np.array([1.8132399004108258e-06,3.0194343342832945e-08,4.808481613954586e-10,7.550027186797983e-12,1.18179823506625e-13])
a_upir=np.array([0.060892450502595924,0.03062604167916817,0.015335676145570262,0.0076706755051601395,0.0038356926037501214])
a_uppg=np.array([1.531133228042791e-06,2.8566384750952173e-08,4.734193680693146e-10,7.519640746452737e-12,1.1810340948877285e-13])

fig1,ax1 = plt.subplots()
plt.loglog(dx,a_orig,'-o',c='g')
#plt.loglog(n_els,a_upir,'-o',c='r')
plt.loglog(dx,a_uppg,'-+',c='b')
#plt.loglog(n_els,dx,'-.k')
plt.loglog(dx,dx*dx*dx*dx*dx*dx,'--k')
#plt.legend(['A','A (IR)','A (PG)','$dx^1$','$dx^6$'])
ax1.set_xticks([0.02,0.04,0.06,0.08,0.1,0.2])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.legend(['$A$','$A_{PG;\Delta t}$','$dx^6$'])
plt.title('$L^2$ error convergence, $p=6$')
plt.xlabel('element width, $\Delta x$')
plt.savefig('l2_convergence_p6_revised.pdf')
plt.show()

############# material form operator: u.GRAD q ###############

# N = 3

a_orig = np.array([0.047934307012048,0.004853326888379379,0.0005802304692904473,7.12831769464754e-05,8.867442047573596e-06])
a_down = np.array([0.04992730035008568,0.00517568775887848,0.0006177387258352896,7.5852517359956e-05,9.434571473282832e-06])

fig1,ax1 = plt.subplots()
plt.loglog(dx,a_orig,'-o',c='g')
plt.loglog(dx,a_down,'-+',c='b')
plt.loglog(dx,dx*dx*dx,'--k')
ax1.set_xticks([0.02,0.04,0.06,0.08,0.1,0.2])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.legend(['$A$',r'$-A_{PG;-\Delta t}^{\top}$','$dx^3$'])
plt.title('$L^2$ error convergence, $p=3$')
plt.xlabel('element width, $\Delta x$')
plt.savefig('l2_convergence_p3_mat_form.pdf')
plt.show()

# N = 6

a_orig = np.array([1.2887232025631434e-05,1.8708510913643899e-06,3.132095458785885e-08,4.988531522100156e-10,7.83386390163029e-12])
a_down = np.array([1.638003547523529e-05,1.8775465828115952e-06,3.137684394511913e-08,4.991156839918155e-10,7.835779220030812e-12])

fig1,ax1 = plt.subplots()
plt.loglog(dx,a_orig,'-o',c='g')
plt.loglog(dx,a_down,'-+',c='b')
plt.loglog(dx,dx*dx*dx*dx*dx*dx,'--k')
ax1.set_xticks([0.02,0.04,0.06,0.08,0.1,0.2])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.legend(['$A$',r'$-A_{PG;-\Delta t}^{\top}$','$dx^6$'])
plt.title('$L^2$ error convergence, $p=6$')
plt.xlabel('element width, $\Delta x$')
plt.savefig('l2_convergence_p6_mat_form.pdf')
plt.show()

