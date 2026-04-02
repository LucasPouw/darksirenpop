import numpy as np
import matplotlib.pyplot as plt
import sys, os

from scipy.special import chebyt
from scipy.integrate import romb

import astropy.units as u

from utils import log10addexp10


COEFFICIENTS = {
    0: np.array([
        [-7.798, +0.145, -0.157],  # c0,0
        [ 1.128, +0.085, -0.081],  # c0,1
        [-0.120, +0.005, -0.006],  # c0,2
    ]),
    
    1: np.array([
        [-17.163, +0.219, -0.226],   # c1,0
        [ -5.512, +0.127, -0.124],   # c1,1
        [  0.593, +0.011, -0.010],   # c1,2
        [ -0.024, +0.00035, -0.00039], # c1,3
    ]),
    
    2: np.array([
        [ -3.223, +0.127, -0.121],   # c2,0
        [ -0.258, +0.047, -0.051],   # c2,1
    ]),
    
    3: np.array([
        [ -2.312, +0.034, -0.032],   # c3,0
        [  0.559, +0.049, -0.045],   # c3,1
        [  3.773, +0.017, -0.016],   # c3,2
        [141.884, +31.521, -3.832],  # c3,3
        [ -0.171, +0.101, -0.116],   # c3,4
    ])
}


# def QLF(M, phi_star, M_star, alpha, beta):
#     """
#     M: absolute magnitude
#     phi_star: amplitude
#     M_star: break magnitude
#     alpha: bright-end slope
#     beta: faint-end slope
#     """
#     return phi_star / (10**(0.4 * (alpha + 1) * (M - M_star)) + 10**(0.4 * (beta + 1) * (M - M_star)))


def F(i, z):
    z = np.atleast_1d(z)
    x = 1 + z
    ci = COEFFICIENTS[i][:,0]
    
    if i != 3:
        Fi_x = np.zeros_like(z)
        for j, cij in enumerate(ci):
            Fi_x += cij * chebyt(j)(x)
        return Fi_x
    
    else:
        c30, c31, c32, c33, c34 = ci
        zeta = np.log10( x / (1 + c32) )
        return c30 + c31 / (10**(c33 * zeta) + 10**(c34 * zeta))


def log10_phi_star_zevo_kulkarni(z):
    return F(0, z)


def M_star_zevo(z):
    return F(1, z)


def alpha_zevo(z):
    return F(2, z)


def beta_zevo(z):
    return F(3, z)


def log10_QLF_kulkarni(M, z):
    """
    M: absolute magnitude
    log10_phi_star: log base 10 of amplitude [mag^-1 cMpc^-3]
    M_star: break magnitude
    alpha: bright-end slope
    beta: faint-end slope
    """
    log10_phi_star = log10_phi_star_zevo_kulkarni(z)
    M_star = M_star_zevo(z)
    alpha = alpha_zevo(z)
    beta = beta_zevo(z)

    term1 = 0.4 * (alpha + 1) * (M - M_star)
    term2 = 0.4 * (beta + 1) * (M - M_star)
    return log10_phi_star - log10addexp10(term1, term2)


COEFFICIENTS_A = {
    'gamma_1': np.array([
                [0.8569, +0.0247, -0.0253],  # a0
                [-0.2614, +0.0162, -0.0164],  # a1
                [0.0200, +0.0011, -0.0011]  # a2
    ]),

    'gamma_2': np.array([
                [2.5375, +0.0177, -0.0187],  # b0
                [-1.0425, +0.0164, -0.0182],  # b1
                [1.1201, +0.0199, -0.0207]  # b2
    ]),

    'log10_L_star': np.array([
                [13.0088, +0.0090, -0.0091],  # c0
                [-0.5759, +0.0018, -0.0020],  # c1
                [0.4554, +0.0028, -0.0027]  # c2
    ]),

    'log10_phi_star': np.array([
                [-3.5426, +0.0235, -0.0209],  # d0
                [-0.3936, +0.0070, -0.0073]  # d1
    ]),
}

COEFFICIENTS_B = {
    'gamma_1': np.array([
                [0.3653, +0.0115, -0.0114],  # a0
                [-0.6006, +0.0422, -0.0417],  # a1
    ]),

    'gamma_2': np.array([
                [2.4709, +0.0163, -0.0169],  # b0
                [-0.9963, +0.0167, -0.0161],  # b1
                [1.0716, +0.0180, -0.0181]  # b2
    ]),

    'log10_L_star': np.array([
                [12.9656, +0.0092, -0.0089],  # c0
                [-0.5758, +0.0020, -0.0019],  # c1
                [0.4698, +0.0025, -0.0026]  # c2
    ]),

    'log10_phi_star': np.array([
                [-3.6276, +0.0209, -0.0203],  # d0
                [-0.3444, +0.0063, -0.0061]  # d1
    ]),
}


def gamma_1_zevo(z, model='A'):
    if model=='A':
        coefficients = COEFFICIENTS_A

        a0, a1, a2 = coefficients['gamma_1'][:,0]
        gamma_1 = a0 * chebyt(0)(1 + z) + a1 * chebyt(1)(1 + z) + a2 * chebyt(2)(1 + z)

    elif model=='B':
        coefficients = COEFFICIENTS_B

        z_ref = 2  # Choice by Shen et al.
        a0, a1 = coefficients['gamma_1'][:,0]
        gamma_1 = np.log10(a0) + a1 * np.log10((1 + z) / (1 + z_ref))
        gamma_1 = 10**gamma_1
    
    else:
        sys.exit(f'Unrecognized model, got: {model}')

    return gamma_1


def gamma_2_zevo(z, model='A'):
    if model=='A':
        coefficients = COEFFICIENTS_A
    elif model=='B':
        coefficients = COEFFICIENTS_B
    else:
        sys.exit(f'Unrecognized model, got: {model}')

    z_ref = 2  # Choice by Shen et al.
    b0, b1, b2 = coefficients['gamma_2'][:,0]
    log10_numerator = np.log10(2 * b0)
    denominator_log10_term1 = b1 * np.log10((1 + z) / (1 + z_ref))
    denominator_log10_term2 = b2 * np.log10((1 + z) / (1 + z_ref))
    log10_denominator = log10addexp10(denominator_log10_term1, denominator_log10_term2)
    return 10**(log10_numerator - log10_denominator)


def log10_L_star_zevo(z, model='A'):
    if model=='A':
        coefficients = COEFFICIENTS_A
    elif model=='B':
        coefficients = COEFFICIENTS_B
    else:
        sys.exit(f'Unrecognized model, got: {model}')
    
    z_ref = 2  # Choice by Shen et al.
    c0, c1, c2 = coefficients['log10_L_star'][:,0]
    log10_numerator = np.log10(2 * c0)
    denominator_log10_term1 = c1 * np.log10((1 + z) / (1 + z_ref))
    denominator_log10_term2 = c2 * np.log10((1 + z) / (1 + z_ref))
    log10_denominator = log10addexp10(denominator_log10_term1, denominator_log10_term2)
    return 10**(log10_numerator - log10_denominator)


def log10_phi_star_zevo(z, model='A'):
    if model=='A':
        coefficients = COEFFICIENTS_A
    elif model=='B':
        coefficients = COEFFICIENTS_B
    else:
        sys.exit(f'Unrecognized model, got: {model}')

    d0, d1 = coefficients['log10_phi_star'][:,0]
    log10_phi_star = d0 * chebyt(0)(1 + z) + d1 * chebyt(1)(1 + z)
    return log10_phi_star


def log10_QLF_shen(log10_L_sollum, z, model='A'):
    """
    log10_L: log10 bolometric luminosity [Lsun]
    log10_phi_star: log base 10 of amplitude [dex^-1 cMpc^-3]
    L_star: break luminosity [Lsun]
    gamma_1: faint-end slope
    gamma_2: bright-end slope
    """
    log10_phi_star = log10_phi_star_zevo(z, model)
    log10_L_star = log10_L_star_zevo(z, model)
    gamma_1 = gamma_1_zevo(z, model)
    gamma_2 = gamma_2_zevo(z, model)

    log10_term1 = gamma_1 * (log10_L_sollum - log10_L_star)
    log10_term2 = gamma_2 * (log10_L_sollum - log10_L_star)

    return log10_phi_star - log10addexp10(log10_term1, log10_term2)


def L2M(log10_Lbol):
    """
    Convert bolometric luminosity in erg/s to absolute AB magnitude at 1450 Angstrom. We use the magnitude-dependention function (Eq. 9) of Runnoe et al. (2012).
    """
    log10_Liso = log10_Lbol - np.log10(0.75)  # Viewing angle correction
    log10_nuLnu1450 = (log10_Liso - 4.74) / 0.91
    log10_Lnu1450 = log10_nuLnu1450 - np.log10((1450 * u.Angstrom).to(u.Hz, equivalencies=u.spectral()).value)

    offset = 2.5 * np.log10( 4 * np.pi * ((10 * u.pc).to(u.cm).value)**2 ) - 48.60
    absmag_ab = lambda log10_Lnu: -2.5 * log10_Lnu + offset
    return absmag_ab(log10_Lnu1450)


def get_n_of_z(qlf, log10_Lbol_thresh):
    '''
    qlf: str, either 'kulkarni', 'shenA' or 'shenB'
    log10_Lbol_thresh: log10 bolometric luminosity threshold in erg/s
    '''
    
    if qlf in ['shenA', 'shenB']:
        if qlf == 'shenA':
            model = 'A'
        else:
            model = 'B'
        log10_L_thresh_solmass = np.log10( (10**log10_Lbol_thresh * u.erg / u.s).to(u.Lsun).value )
        lum_integrate_axis = np.linspace(log10_L_thresh_solmass, 100, int(1024)+1)
        n_of_z = lambda z: romb(10**log10_QLF_shen(lum_integrate_axis[:,np.newaxis], z=z[np.newaxis,:], model=model), dx=np.diff(lum_integrate_axis)[0], axis=0)

    elif qlf == 'kulkarni':
        magthresh = L2M(log10_Lbol_thresh)
        mag_integrate_axis = np.linspace(-100, magthresh, int(1024)+1)
        n_of_z = lambda z: romb(10**log10_QLF_kulkarni(mag_integrate_axis[:,np.newaxis], z=z[np.newaxis,:]), dx=np.diff(mag_integrate_axis)[0], axis=0)
    
    else:
        sys.exit(f'Do not recognize QLF: {qlf}')
    
    return n_of_z



if __name__ == '__main__':

    zz = np.linspace(0, 7, 1000)
    plt.figure(figsize=(8,6))
    plt.plot(zz, gamma_1_zevo(zz, model='A'), color='purple', linewidth=3)
    plt.plot(zz, gamma_1_zevo(zz, model='B'), color='hotpink', linewidth=3)
    plt.xlim(0, 7)
    plt.ylim(-0.1, 1.7)
    plt.ylabel('Faint-end slope')
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(zz, gamma_2_zevo(zz, model='A'), color='purple', linewidth=3)
    plt.plot(zz, gamma_2_zevo(zz, model='B'), color='hotpink', linewidth=3)
    plt.xlim(0, 7)
    plt.ylim(0.9, 2.9)
    plt.ylabel('Bright-end slope')
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(zz, log10_L_star_zevo(zz, model='A'), color='purple', linewidth=3)
    plt.plot(zz, log10_L_star_zevo(zz, model='B'), color='hotpink', linewidth=3)
    plt.xlim(0, 7)
    plt.ylim(10.3, 13.7)
    plt.ylabel(r'$\log_{10} L_{*} [L_{\odot}]$')
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(zz, log10_phi_star_zevo(zz, model='A'), color='purple', linewidth=3)
    plt.plot(zz, log10_phi_star_zevo(zz, model='B'), color='hotpink', linewidth=3)
    plt.xlim(0, 7)
    plt.ylim(-6.3, -3.1)
    plt.ylabel(r'$\log_{10} \phi_{*} [\mathrm{dex^{-1} \, cMpc^{-3}}]$')
    plt.show()

    LL = np.geomspace(1e42, 1e50) * u.erg / u.s
    log10_LL_sollum = np.log10( LL.to(u.Lsun).value )

    zfid = 3

    plt.figure(figsize=(8,6))
    plt.plot(np.log10(LL.value), log10_QLF_shen(log10_LL_sollum, z=zfid, model='A'), color='purple', linewidth=2)
    plt.plot(np.log10(LL.value), log10_QLF_shen(log10_LL_sollum, z=zfid, model='B'), color='hotpink', linewidth=2)
    # plt.semilogx()
    plt.xlim(42.5, 50.25)
    plt.ylim(-11.1, -2.4)
    # plt.xticks([])
    plt.show()


    # z_eval = 0.72 # 0.31#
    # plt.figure()
    # # plt.plot(np.linspace(-30, 0, 1000), np.log10( QLF(np.linspace(-30, 0, 1000), phi_star=10**(-5.72), M_star=-21.30, alpha=-2.74, beta=-1.07) ), label='In bin')
    # plt.plot(np.linspace(-30, 0, 1000), np.log10( QLF(np.linspace(-30, 0, 1000), phi_star=10**(-6.57), M_star=-24.21, alpha=-3.55, beta=-1.89) ), label='In bin')
    # plt.plot(np.linspace(-30, 0, 1000), log10_QLF_kulkarni(np.linspace(-30, 0, 1000), z=z_eval), label='All z')
    # plt.legend()
    # plt.show()

    z_ax = np.linspace(0, 7, 1000)
    plt.figure(figsize=(8,6))
    plt.plot(z_ax, log10_phi_star_zevo_kulkarni(z_ax))
    plt.xlim(0, 7)
    plt.ylim(-12, -5)
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(z_ax, M_star_zevo(z_ax))
    plt.xlim(0, 7)
    plt.ylim(-32, -20)
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(z_ax, alpha_zevo(z_ax))
    plt.xlim(0, 7)
    plt.ylim(-7, -1)
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(z_ax, beta_zevo(z_ax))
    plt.xlim(0, 7)
    plt.ylim(-3, 0)
    plt.show()
