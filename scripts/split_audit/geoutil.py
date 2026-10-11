import numpy as np
A = 6378137.0; F = 1/298.257223563
def latlon_to_utm(lat, lon, zone):
    """WGS84 -> UTM (northern hemisphere), Krueger series (mm accuracy)."""
    lat = np.radians(np.asarray(lat, float)); lon = np.radians(np.asarray(lon, float))
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    k0 = 0.9996; n = F / (2 - F)
    Ahat = A / (1 + n) * (1 + n**2/4 + n**4/64)
    al = [None, n/2 - 2*n**2/3 + 5*n**3/16, 13*n**2/48 - 3*n**3/5, 61*n**3/240]
    t = np.sinh(np.arctanh(np.sin(lat)) - 2*np.sqrt(n)/(1+n)*np.arctanh(2*np.sqrt(n)/(1+n)*np.sin(lat)))
    dl = lon - lon0
    xi = np.arctan2(t, np.cos(dl)); eta = np.arctanh(np.sin(dl)/np.sqrt(1+t**2))
    E = eta.copy(); N = xi.copy()
    for j in range(1, 4):
        E += al[j]*np.cos(2*j*xi)*np.sinh(2*j*eta)
        N += al[j]*np.sin(2*j*xi)*np.cosh(2*j*eta)
    return 500000 + k0*Ahat*E, k0*Ahat*N
