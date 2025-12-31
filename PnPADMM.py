import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from numpy.fft import fft2, ifft2, ifftshift
from numpy.random import randn
from skimage import io, color, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from bm3d import bm3d
from scipy.sparse.linalg import cg, LinearOperator




def gaussian_kernel(size=25, sigma=1):
    """Create normalized 2D Gaussian kernel."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= kernel.sum()
    return kernel


def blur_and_noise(img, kernel, noise_std=1/255):
    """Apply Gaussian blur and additive white noise."""
    y_blur = scipy.signal.convolve2d(img, kernel, mode='same', boundary='symm')
    y = y_blur + noise_std * np.random.randn(*img.shape)
    return np.clip(y, 0, 1)




def pnp_admm_deconv(y, h, gt=None, lam=1e-2, rho0=1e-2,
                    gamma=1.2, max_iter=50, show_log=True):
    """
    Plug-and-Play ADMM for image deconvolution using BM3D.
    f(x) = ||y - h*x||^2
    Denoiser acts as proximal of g(x).
    """

    pad = h.shape[0] // 2
    y_pad = np.pad(y, pad, mode='symmetric')

    H = fft2(h, s=y_pad.shape)
    Hc = np.conj(H)

   
    x = y_pad.copy()
    v = y_pad.copy()
    u = np.zeros_like(y_pad)
    rho = rho0
    psnrs = []

    for k in range(max_iter):
        
        num = Hc * fft2(y_pad) + rho * fft2(v - u)
        den = np.abs(H)**2 + rho
        x = np.real(ifft2(num / den))

       
        sigma = np.sqrt(lam / rho)
        v = bm3d(x + u, sigma_psd=sigma)

        
        u = u + (x - v)

        
        rho *= gamma

        
        if gt is not None and ((k + 1) % 10 == 0 or k == 0):
            x_crop = x[pad:-pad, pad:-pad]
            cur_psnr = psnr(gt, np.clip(x_crop, 0, 1))
            psnrs.append(cur_psnr)
            if show_log:
                print(f"[Deconv] Iter {k+1:03d}: rho={rho:.3e}, sigma={sigma:.4f}, PSNR={cur_psnr:.2f} dB")

    # Crop padded borders
    x_final = np.clip(x[pad:-pad, pad:-pad], 0, 1)
    return x_final, psnrs




def pnp_admm_cs(gt, m_ratio=4096, lam=1e-2, rho0=1e-2,
                gamma=1.2, max_iter=50, show_log=True):
    """
    Plug-and-Play ADMM for Compressive Sensing Reconstruction.
    Solves: min_x ||y - A x||^2 + lambda g(x)
    using BM3D as denoiser.
    """

    nx, ny = gt.shape
    N = nx * ny

  
    A = randn(m_ratio, N) / np.sqrt(m_ratio)
    AT = A.T

    x_true = gt.ravel()
    y = A @ x_true + 0.01 * randn(m_ratio)

    
    x = AT @ y
    v = x.copy()
    u = np.zeros_like(x)
    rho = rho0
    psnrs = []

    for k in range(max_iter):
        
        rhs = AT @ y + rho * (v - u)

        def AHA(z): return AT @ (A @ z) + rho * z
        linop = LinearOperator((N, N), matvec=AHA)

        x, _ = cg(linop, rhs, x0=x, maxiter=100, rtol=1e-6)

        
        x_img = x.reshape(gt.shape)
        sigma = np.sqrt(lam / rho)
        v_img = bm3d(x_img + u.reshape(gt.shape), sigma_psd=sigma)
        v = v_img.ravel()

        
        u = u + (x - v)

        
        rho *= gamma

        
        if ((k + 1) % 10 == 0 or k == 0):
            rec = np.clip(x.reshape(gt.shape), 0, 1)
            cur_psnr = psnr(gt, rec)
            psnrs.append(cur_psnr)
            if show_log:
                print(f"[CS] Iter {k+1:03d}: rho={rho:.3e}, sigma={sigma:.4f}, PSNR={cur_psnr:.2f} dB")

    rec = np.clip(x.reshape(gt.shape), 0, 1)
    return rec, psnrs




if __name__ == "__main__":

   
    url = "ADMM.png"  
    gt = io.imread(url)
    if gt.ndim == 3:
        if gt.shape[2] == 4:
            gt = gt[..., :3]
        gt = color.rgb2gray(gt)
    gt = img_as_float(gt)
    gt = gt[:128, :128]  # resize for faster testing
    print("Loaded image shape:", gt.shape)

   
    for s_blur in [1, 3]:
        h = gaussian_kernel(25, s_blur)
        y = blur_and_noise(gt, h)
        print(f"\n=== Deconvolution for blur sigma={s_blur} ===")
        restored, psnrs = pnp_admm_deconv(y, h, gt,
                                          lam=1e-2, rho0=1e-2,
                                          gamma=1.2, max_iter=60)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.imshow(gt, cmap='gray'); plt.title('Original'); plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(y, cmap='gray'); plt.title(f'Blur σ={s_blur}'); plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(restored, cmap='gray')
        plt.title(f'Restored (PSNR={psnr(gt, restored):.2f} dB)'); plt.axis('off')
        plt.tight_layout(); plt.show()

   
    for m in [4096, 8192]:
        print(f"\n=== Compressive Sensing (m={m}) ===")
        rec, psnrs = pnp_admm_cs(gt, m_ratio=m,
                                 lam=1e-2, rho0=1e-2,
                                 gamma=1.2, max_iter=50)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1); plt.imshow(gt, cmap='gray'); plt.title('Original'); plt.axis('off')
        plt.subplot(1, 2, 2); plt.imshow(rec, cmap='gray')
        plt.title(f'CS (m={m}) PSNR={psnr(gt, rec):.2f} dB'); plt.axis('off')
        plt.tight_layout(); plt.show()
