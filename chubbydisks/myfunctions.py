#
# ALL FUNCTIONS
# Cristiano Longarini
#
# Units:
# - distances in au
# - mass in msun
# - velocities in km/s
#

import matplotlib.pyplot as plt
import scipy
import numpy as np
import math
import matplotlib.image as mpimg
from scipy.interpolate import griddata
from scipy import special
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.integrate import simps
from astropy import constants as const
from scipy.interpolate import griddata

G = 4.30091e-3 * 206265 

def omega(ms,r):
    
    '''Keplerian frequency in [Hz]
    ms = mass of the central object [msun]
    r = radius [au]'''
   
    return np.sqrt(G*ms/r**3)



def zeta(r1,r,z):
    
    return np.sqrt((4*r1*r )/ ((r+r1)**2 +z**2))



def sigmain(p, rin, rout, md):
    
    x = rout/rin
    return ((2+p)*md)/(2*np.pi*rin**2) * (x**(2+p) -1)**(-1)



def sigma(p, rin, rout ,md, r):
    
    '''Surface density of the disc in [Msun/au^2]
    p = power law index of the density profile. \Sigma \propto r^(p)
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    md = mass of the disc [msun]
    r = radius [au]'''
    
    return sigmain(p,rin,rout,md) * (r/rin)**(+p)



def integrand(r1, r, z, md, p, rin, rout):
    
    zet = zeta(r1,r,z)
    kappa = scipy.special.ellipk(zet)
    ellip = scipy.special.ellipe(zet)
    
    return (kappa - 1/4 * (zet**2 /(1-zet**2)) * (r1/r -r/r1+ 
           (z**2)/(r*r1))*ellip)*np.sqrt(r1/r)*zet*sigma(p,rin,rout,md,r1)



def veldisc(r, z, md, p, rin, rout):
    
    def expint(r,z,md,p,rin,rout):
        return quad(integrand, 1/2 * rin, 2*rout, args=(r,z,md,p,rin,rout))[0]
    vec_expint = np.vectorize(expint)
    
    return G  * vec_expint(r,z,md,p,rin,rout)



def basicspeed(r, z, md, p, rin, rout, ms):
    
    '''Rotation curve of a self gravitating disc [km/s]
    r = radius [au]
    z = height [au] (midplane z=1e-3)
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    ms = mass of the central object [msun]'''
    
    return np.sqrt( (G*ms/r) + veldisc(r,z,md,p,rin,rout))



def q(ms, md, p, rin, rout, r):
    
    '''Disc to star mass ratio for a Q=1 disc
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    r = radius [au]'''
    
    q_ext = md / ms
    
    return q_ext * (rout/rin)**(-2-p) * (r/rin)**(2+p)



def ura(ms, md, p, m, chi, beta, rin, rout, r):
    
    '''Module of the radial velocity perturbation [km/s]
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    r = radius [au]'''
    
    return 8 * m * chi * beta**(-1/2) * q(ms, md, p, rin, rout, r)**2 * omega(ms,r) * r



def upha(ms, md, p, m, chi, beta, rin, rout, r):
    
    '''Module of the azimuthal velocity perturbation [km/s]
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    r = radius [au]'''
    
    return - (m * chi * beta**(-1/2))  * q(ms, md, p, rin, rout, r) * omega(ms,r) * r



def ur(grid_radius, grid_angle, ms, md, p, m, chi, beta, rin, rout, alpha, off):
    
    '''2D radial velocity perturbation [km/s] in polar coordinates
    grid_radius = radial grid [au]
    grid_angle = azimuthal grid [-np.pi,np.pi] [rad]
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]'''
    
    return - ura(ms, md, p, m, chi, beta, rin, rout, grid_radius)  * np.sin(
        m * grid_angle + m/np.tan(alpha) * np.log(grid_radius) + off)



def uph(grid_radius, grid_angle, ms, md, p, m, chi, beta, rin, rout, alpha, off):
    
    '''2D azimuthal velocity perturbation [km/s] in polar coordinates
    grid_radius = radial grid [au]
    grid_angle = azimuthal grid [-np.pi,np.pi] [rad]
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]'''
    
    x = np.linspace(rin,rout,grid_radius.shape[0])
    phase = m / np.tan(alpha)  * np.log(x)
    an = np.linspace(-np.pi,np.pi,grid_radius.shape[1])
    bs = basicspeed(x, 0.001, md, p, rin, rout, ms)
    vec = np.zeros([grid_radius.shape[0],grid_radius.shape[1]])
    vp1 = upha(ms, md, p, m, chi, beta, rin, rout, rin)
    for i in range(grid_radius.shape[1]):
        vec[:,i] = bs[:] - vp1* x**(3/2 + p) /rin * np.sin(m*an[i] + phase[:] + off)
        
    return vec



def momentone(grid_radius, grid_angle, ms, md, p, m, chi, beta, rin, rout, alpha, incl, off):
    
    '''Moment one map / projected velocity field towards the line of sight [km/s]
    grid_radius = radial grid [au]
    grid_angle = azimuthal grid [-np.pi,np.pi] [rad]
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]
    incl = inclination angle [rad]
    NB: The moment one map is given in polar coordinates, the disk is face on and the observer is rotated by an angle incl'''
    
    return uph(grid_radius, grid_angle, ms, md, p, m, chi, beta, rin, rout, alpha, off) * np.cos(
        grid_angle) * np.sin(incl) + ur(grid_radius, grid_angle, ms, md, p, m, chi, beta,
        rin, rout, alpha, off) * np.sin(grid_angle) * np.sin(incl)



def momentone_keplerian(grid_radius, grid_angle, ms, incl):
    
    '''Keplerian moment one map / projected velocity field towards the line of sight [km/s]
    grid_radius = radial grid [au]
    grid_angle = azimuthal grid [-np.pi,np.pi] [rad]
    ms = mass of the central object [msun]
    incl = inclination angle [rad]'''
    
    vk = (omega(ms, grid_radius) * grid_radius)
    
    return vk * np.cos(grid_angle) * np.sin(incl)



def perturbed_sigma(grid_radius, grid_angle, p, rin, rout ,md, beta, m, alpha, pos):
    
    '''Spiral-perturbed surface density [msun / au^2]
    grid_radius = radial grid [au]
    grid_angle = azimuthal grid [-np.pi,np.pi] [rad]
    p = power law index of the density profile. \Sigma \propto r^(p)
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    md = mass of the disc [msun]
    m = number of the spiral arms
    alpha = pitch angle of the spiral [rad]
    pos = angle of the spiral within the disc [rad]'''
    
    return sigma(p, rin, rout ,md, grid_radius) + sigma(p, rin, rout ,md, grid_radius) * beta**(-1/2) * np.sin(m * 
    grid_angle + m/np.tan(alpha) * np.log(grid_radius) + pos)



def get_masses(statement, rot_curve, radii, ms, z, starmin, starmax, discmin, discmax, p, n):
    
    '''Very simple algorithm that optimises the mass of the disc+star / disc / star from the rotation curve
    The output is the 1D / 2D array containing the std deviation.
    statement = -1 if you want mass of the disc and mass of the star,
                 0 if you want only the mass of the disc
                 1 if you want only the mass of the star
    rot_curve = vector of the rotation curve [km/s]
    radii = vector of the radii [au]
    ms = mass of the central object [msun], if you want to find it simply put 0
    z = height [au] (for midplane put z=1e-3)
    starmin = minimum value of the mass of the star [msun], if you want to find only md put 0 
    starmax = maximum value of the mass of the star [msun], if you want to find only md put 0 
    discmin = minimum value of the mass of the disc [msun], if you want to find only ms put 0 
    discmax = minimum value of the mass of the disc [msun], if you want to find only ms put 0 
    p = power law index of the density profile. \Sigma \propto r^(p)
    n = number of the point within the interval of research'''
    
    if (statement == -1): #compute both star and disc mass
        a = np.linspace(starmin, starmax, n)
        b = np.linspace(discmin, discmax, n)
        vec = np.zeros([n,n])
        for i in range (n):
            for j in range (n):
                vec[i,j] = np.sum( (rot_curve - basicspeed(radii, z, b[j], p, radii[0], 
                            radii[len(radii)-1], a[i]))**2 )

        
    if (statement == 0): #compute only disc mass
        b = np.linspace(discmin, discmax, n)
        vec = np.zeros(n)
        for i in range(n):
            vec[i] = np.sum( (rot_curve - basicspeed(radii, z, b[j], p, radii[0], radii[len(radii)-1], ms))**2 )
        
        
    if (statement == 1): #compute only star mass
        a = np.linspace(starmin, starmax, n)
        vec = np.zeros(n)
        for i in range(n):
            vec[i] = np.sum( (rot_curve - omega(a[i], radii) * radii )**2)
    

    else :
        print('Error in statement, you must choose between -1, 0 and 1.')
    
    return vec



def amplitude_central_channel(grid_radius, grid_angle, ms, md, p, m, chi, beta, rin, rout, alpha, incl):
    
    '''Amplitude of the central channel of the moment one map (v_obs=v_syst) [rad]
    grid_radius = radial grid [au]
    grid_angle = azimuthal grid [-np.pi,np.pi] [rad]
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]
    incl = inclination angle [rad]
    '''
    
    m1 = momentone(grid_radius, grid_angle, ms, md, p, m, chi, beta, rin, rout, alpha, incl)
    wigg = np.zeros(grid_radius.shape[0])
    
    for i in range(grid_radius.shape[0]):
        for j in range(grid_angle.shape[0]-1):
            if (m1[i,j] < 0 and m1[i,j+1] > 0):
                wigg[i] = grid_angle[0,:][j]
    
    return np.std(wigg)




def urC(gx, gy, ms, md, p, m, chi, beta, rin, rout, alpha, off):
    
    '''2D radial velocity perturbation [km/s] in polar coordinates
    gx = x grid [au]
    gy = y grid [au] 
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]'''
    
    
    grid_radius = np.sqrt(gx**2 + gy**2)
    car = np.linspace(-rout,rout,gx.shape[0])
    grid_angle = np.zeros([len(car),len(car)])
    for i in range(len(car)):
        for j in range(len(car)):
            grid_angle[i,j] = math.atan2(car[i], car[j])
    return - ura(ms, md, p, m, chi, beta, rin, rout, grid_radius)  * np.sin(
        m * grid_angle + m/np.tan(alpha) * np.log(grid_radius) + off)



def uphC(gx, gy, ms, md, p, m, chi, beta, rin, rout, alpha, off):
    
    '''2D azimuthal velocity perturbation [km/s] in polar coordinates
    gx = x grid [au]
    gy = y grid [au] 
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]'''
    
    # x = np.linspace(np.min(grid_radius),np.max(grid_radius),grid_radius.shape[0]/2)
    #phase = m / np.tan(alpha)  * np.log(x)
    #an = np.linspace(-np.pi,np.pi,grid_radius.shape[1])
    #bs = basicspeed(x, 0.001, md, p, rin, rout, ms)
    #vec = np.zeros([grid_radius.shape[0],grid_radius.shape[1]])
    #vp1 = upha(ms, md, p, m, chi, beta, rin, rout, rin)
    #for i in range(grid_radius.shape[1]):
    #   vec[:,i] = bs[:] - vp1* x**(3/2 + p) /rin * np.sin(m*an[i] + phase[:] + off)
    
    grid_radius = np.sqrt(gx**2 + gy**2)
    car = np.linspace(-rout,rout,gx.shape[0])
    grid_angle = np.zeros([len(car),len(car)])
    for i in range(len(car)):
        for j in range(len(car)):
            grid_angle[i,j] = math.atan2(car[i], car[j])
    
    radii = np.linspace(np.min(grid_radius), np.max(grid_radius), 1000)
    rc = basicspeed(radii, 1e-3, md, p, rin, rout, ms)
    vec = - upha(ms, md, p, m, chi, beta, rin, rout, grid_radius) + rc[
        ((grid_radius - np.min(grid_radius))/(radii[1] - radii[0])).astype(int)] 
    return vec



def momentoneC(gx, gy, ms, md, p, m, chi, beta, rin, rout, alpha, incl, off):
    
    '''Moment one map / projected velocity field towards the line of sight [km/s]
    gx = x grid [au]
    gy = y grid [au] 
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]
    incl = inclination angle [rad]
    NB: The moment one map is given in polar coordinates, the disk is face on and the observer is rotated by an angle incl'''
    
    grid_radius = np.sqrt(gx**2 + gy**2)
    car = np.linspace(-rout,rout,gx.shape[0])
    grid_angle = np.zeros([len(car),len(car)])
    for i in range(len(car)):
        for j in range(len(car)):
            grid_angle[i,j] = math.atan2(car[i], car[j])
    
    M1 = uphC(gx, gy, ms, md, p, m, chi, beta, rin, rout, alpha, off) * np.cos(
        grid_angle) * np.sin(incl) + urC(gx, gy, ms, md, p, m, chi, beta,
        rin, rout, alpha, off) * np.sin(grid_angle) * np.sin(incl)
    for i in range(M1.shape[0]):
        for j in range(M1.shape[0]):
            if(grid_radius[i,j] > rout):
                M1[i,j] = -np.inf
    return M1
