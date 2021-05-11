from chubbydisks import myfunctions


def test_omega():
    assert myfunctions.omega(1,1) == 29.78468064542576
    
    
def test_zeta():
    assert myfunctions.zeta(1,1,1) == 0.8944271909999159
    

def test_sigmain():
    assert myfunctions.sigmain(-1,1,10,1) == 0.017683882565766147
    

def test_sigma():
    assert myfunctions.sigma(-1,1,10,1,1) == 0.017683882565766147
    
    
def test_integrand():
    assert myfunctions.integrand(1,1,0.001,1,-1,1,10) == 0.1473733953553126
    
    
def test_basicspeed():
    assert myfunctions.basicspeed(10,0.001,1,-1,1,100,1) == 10.004681334777239