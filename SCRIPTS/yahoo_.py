# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 13:01:51 2018

@author: kennedy
"""
import pandas as pd
import numpy as np
import time
from datetime import datetime
from os import chdir
from selenium import webdriver
#path = 'D:\\GIT PROJECT\\ERIC_PROJECT101\\FREELANCE_KENNETH\\DATASET'
path = 'C:\\Users\kennedy\\Desktop\\'
#chdir(path)

start_date = pd.Timestamp(2010, 12, 29)
end_date = datetime.today().date()
stock_name = ['500UN.MX', 'AEMEN.MX', 'AGGY.MX', 'AGKN.MX', 'AGRONN.MX', 'AIQ.MX',
              'AME.MX', 'ASIX.MX', 'ATLN.MX', 'AUEMN.MX', 'AUSAUWN.MX', 'AVN.MX', 
              'AXISCK12.MX', 'AZTECAA.MX', 'AZTECADA.MX', 'AZTECADL.MX', 'BAERN.MX', 
              'BBAX.MX', 'BBCA.MX', 'BBEU.MX', 'BBT.MX', 'BELLN.MX', 'BFIT.MX', 
              'BOLNVN.MX', 'BTEC.MX', 'C50N.MX', 'CANAN.MX', 'CATH.MX', 'CBUS5AN.MX', 
              'CBUSACN.MX', 'CCAUN.MX', 'CEMEXA.MX', 'CEMEXB.MX', 'CETETRCIS.MX', 'CEU2N.MX', 
              'CEU3N.MX', 'CEUSN.MX', 'CG1N.MX', 'CHAU.MX', 'CINDN.MX', 'CJ1YN.MX', 'CLNXN.MX', 
              'CMUN.MX', 'CNEN.MX', 'CNYAN.MX', 'CNYAU.MX', 'COPX.MX', 'CORPTRCIS.MX', 'CP9UN.MX', 
              'CRI.MX', 'CRUDN.MX', 'CSKRN.MX', 'CSRUN.MX', 'CSUSN.MX', 'CTTN.MX', 'CTXS.MX', 'CV9N.MX', 
              'CWI.MX', 'D5BLN.MX', 'DDBI.MX', 'DDWM.MX', 'DEMRN.MX', 'DGRAN.MX', 'DLPHN.MX', 'DMRM.MX', 
              'DMRS.MX', 'DRIV.MX', 'DRN.MX', 'DRVV.MX', 'DX2GN.MX', 'DX2XN.MX', 'DXETN.MX', 'DXGYN.MX', 
              'DXJF.MX', 'DXJZN.MX', 'DYLS.MX', 'DZK.MX', 'EATN.MX', 'EDBI.MX', 'EDOARDOB.MX', 'EFNL.MX', 
              'EGUSASN.MX', 'ELUN.MX', 'EMLC1N.MX', 'EMMUSCN.MX', 'EMUAAN.MX', 'ENDPN.MX', 'ENGI.MX', 'EPIEN.MX', 
              'EPS.MX', 'ERY.MX', 'ETLN.MX', 'EUXL.MX', 'EXO1N.MX', 'EXT.MX', 'EXXUN.MX', 'EZEN.MX', 'EZM.MX', 
              'FBZ.MX', 'FECN.MX', 'FEMUN.MX', 'FEUZ.MX', 'FEXUN.MX', 'FJPUN.MX', 'FLBR.MX', 'FLCA.MX', 'FLCH.MX', 
              'FLGR.MX', 'FLIN.MX', 'FLKR.MX', 'FLN.MX', 'FLTW.MX', 'FPXUN.MX', 'FRT.MX', 'FTCS.MX', 'FTXO.MX', 
              'GAAAN.MX', 'GAGUN.MX', 'GASIN.MX', 'GDX1N.MX', 'GDXJ1N.MX', 'GEUPECB.MX', 'GHYB.MX', 'GIGB.MX', 
              'GMACMAB.MX', 'GNK.MX', 'GOEX.MX', 'GSANBORB-.MX', 'GSJN.MX', 'GSJY.MX', 'GSSC.MX', 'GVIP.MX', 
              'HEEN.MX', 'HEION.MX', 'HIKN.MX', 'HIMEXSAB.MX', 'HLEN.MX', 'HMBN.MX', 'HUBNN.MX', 'HYGUN.MX', 
              'HYLAN.MX', 'HYLDN.MX', 'HYZD.MX', 'IAUPN.MX', 'IAUSN.MX', 'IBDH.MX', 'IBDK.MX', 'IBDM.MX', 'IBDN.MX', 
              'IBDO.MX', 'IBDP.MX', 'IBDQ.MX', 'IBDR.MX', 'IBDS.MX', 'IDAPN.MX', 'IDFXN.MX', 'IDP6N.MX', 'IDTKN.MX', 
              'IDTMN.MX', 'IDUPN.MX', 'IDWPN.MX', 'IEMBN.MX', 'IEMLN.MX', 'IEQUN.MX', 'IESZN.MX', 'IGEAN.MX', 'IGLAN.MX', 
              'IGLON.MX', 'IGSUN.MX', 'ILTB.MX', 'IMBSN.MX', 'INDVN.MX', 'INFON.MX', 'INFR.MX', 'INGEALB.MX', 'INTRUMN.MX', 
              'IOCN.MX', 'IPE.MX', 'IRSANN.MX', 'ISFDN.MX', 'IVNN.MX', 'IVVPESOIS.MX', 'JHGN.MX', 'JMBA.MX', 'JPEH.MX', 
              'JPEU.MX', 'JPIH.MX', 'JPME.MX', 'JPNAN.MX', 'JPSE.MX', 'KBA.MX', 'KPNN.MX', 'KRMA.MX', 'KTGN.MX', 'LACOMERU1.MX', 
              'LACOMERUB.MX', 'LASEG.MX', 'LBJ.MX', 'LENB.MX', 'LIVEPOLC-.MX', 'LNAN.MX', 'LNG.MX', 'LNGR.MX', 'LOWC.MX', 
              'LQDHN.MX', 'LVHD.MX', 'LVHE.MX', 'LVHI.MX', 'LWCUN.MX', 'M10TRACIS.MX', 'M5TRACISH.MX', 'MAERSKBN.MX', 'MATASN.MX', 
              'MBWSN.MX', 'MCF.MX', 'MDYV.MX', 'MEXMTUMIS.MX', 'MEXRISKIS.MX', 'MFRISCOA-.MX', 'MIFELO.MX', 'MILN.MX', 'MMKN.MX', 
              'MOAT1N.MX', 'MRWN.MX', 'MWRDN.MX', 'NAFTRAC.N.MX', 'NDIAN.MX', 'NIKONN.MX', 'NSRGYN.MX', 'NTGYN.MX', 'PANASN.MX', 'PAVE.MX', 
              'PAYC.MX', 'PEOLES.MX', 'PFCN.MX', 'PHGN.MX', 'PRSN.MX', 'PUTW.MX', 'PXUS.MX', 'QBINDUSA.MX', 'QBINDUSB.MX', 'QEFA.MX', 
              'QEMM.MX', 'QUS.MX', 'RASSINIB.MX', 'RASSINIC.MX', 'RASSINICP.MX', 'RDVY.MX', 'RENN.MX', 'RHHBYN.MX', 'RUSS.MX', 'RWO.MX', 
              'RWR.MX', 'RWX.MX', 'SANLUISA.MX', 'SANLUISB.MX', 'SANLUISC.MX', 'SANLUISCP.MX', 'SAVIAA.MX', 'SBEMAN.MX', 'SDG.MX', 'SFERN.MX', 
              'SHE.MX', 'SMEZ.MX', 'SMRUN.MX', 'SOFTN.MX', 'SOIL.MX', 'SPAB.MX', 'SPIB.MX', 'SPLB.MX', 'SPSM.MX', 'SPTL.MX', 'SPTS.MX', 'SPYD.MX', 
              'SRETN.MX', 'SUSMN.MX', 'SX7PEXN.MX', 'SXNPEXN.MX', 'SXOPEXN.MX', 'SXRPEXN.MX', 'TDG.MX', 'TEKCHEMA.MX', 'TENCHN.MX', 'TEPN.MX', 
              'TIN.MX', 'TIP1AN.MX', 'TL5N.MX', 'TLEVICPO.MX', 'TLEVISAD.MX', 'TOYTN.MX', 'TPXYN.MX', 'UBBBN.MX', 'UCAPN.MX', 'UDBI.MX', 'UDITRACIS.MX', 
              'UKGBPBN.MX', 'URA.MX', 'USFMDN.MX', 'UT1USN.MX', 'VAHNN.MX', 'VDCPN.MX', 'VDETN.MX', 'VDEVN.MX', 'VDTYN.MX', 'VDUCN.MX', 'VDXXN.MX', 
              'VECPN.MX', 'VETN.MX', 'VETYN.MX', 'VEUDN.MX', 'VGLT.MX', 'VHYDN.MX', 'VUSDN.MX', 'VWSN.MX', 'WCOAN.MX', 'WEIRN.MX', 'WIAUN.MX', 'WLTG.MX', 
              'WPPN.MX', 'WSMLN.MX', 'WSUN.MX', 'XAR.MX', 'XBAUN.MX', 'XCADN.MX', 'XD5DN.MX', 'XDN0N.MX', 'XDNDN.MX', 'XDUDN.MX', 'XDWDN.MX', 'XESPN.MX', 
              'XG7UN.MX', 'XLT.MX', 'XMJDN.MX', 'XMKON.MX', 'XMMEN.MX', 'XMTDN.MX', 'XMUJN.MX', 'XMVEN.MX', 'XMVUN.MX', 'XPXDN.MX', 'XTN.MX', 
              'XUSDN.MX', 'ZALN.MX', 'ZBH.MX', 'ZJPN.MX']

def dwn_yahoo_(path, start, end, stock_):
  date = [start_date, end_date]
  date_epoch = pd.DatetimeIndex(date)
  date_epoch = date_epoch.astype(np.int64) // 10**9
  
  chrom_options_ = webdriver.ChromeOptions()
  
  prefer_ = {'download.default_directory': path,
             'profile.default_content_settings.popups': 0,
             'directory_upgrade': True}
  
  chrom_options_.add_experimental_option('prefs',prefer_)
  
  
  for ii in stock_name:
    try:
      yahoo_page_ = 'https://finance.yahoo.com/quote/{}/history?period1={}&period2={}&interval=1d&filter=history&frequency=1d'.format(ii, date_epoch[0], date_epoch[1])
      driver = webdriver.Chrome("C:/chromedriver.exe", chrome_options = chrom_options_)
    #  driver.minimize_window()
      driver.get(yahoo_page_)
      time.sleep(2)
      driver.find_element_by_css_selector('.btn.primary').click()
      time.sleep(2)
      driver.find_element_by_css_selector('.Fl\(end\).Mt\(3px\).Cur\(p\)').click()
      time.sleep(10)
      driver.close()
    except:
      pass
    
  
dwn_yahoo_(path, start_date, end_date, stock_name)