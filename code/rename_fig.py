
import os

def renaming(src_path,dst_path,sub):
    filelist=os.listdir(src_path)
    print(filelist)


    if sub==1:
        for item in filelist:
            src = os.path.join(os.path.abspath(src_path), item)
            if item=='stimu_0.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(0) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_1.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(1) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_2.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(2) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_3.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(3) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_4.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(4) + '.jpg')  
                os.rename(src, dst)
            if item=='stimu_5.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(5) + '.jpg')  
                os.rename(src, dst) 
            if item=='stimu_6.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(6) + '.jpg')
                os.rename(src, dst) 
            if item=='stimu_7.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(7) + '.jpg') 
                os.rename(src, dst) 

    if sub==2:
        for item in filelist:
            src = os.path.join(os.path.abspath(src_path), item)
            if item=='stimu_0.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(1) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_1.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(5) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_2.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(0) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_3.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(3) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_4.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(6) + '.jpg') 
                os.rename(src, dst) 
            if item=='stimu_5.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(2) + '.jpg')  
                os.rename(src, dst) 
            if item=='stimu_6.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(7) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_7.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(4) + '.jpg')  
                os.rename(src, dst)

    if sub==3:
        for item in filelist:
            src = os.path.join(os.path.abspath(src_path), item)
            if item=='stimu_0.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(1) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_1.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(7) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_2.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(2) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_3.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(4) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_4.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(5) + '.jpg') 
                os.rename(src, dst) 
            if item=='stimu_5.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(3) + '.jpg') 
                os.rename(src, dst)  
            if item=='stimu_6.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(6) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_7.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(0) + '.jpg')  
                os.rename(src, dst)

    if sub==4:
        for item in filelist:
            src = os.path.join(os.path.abspath(src_path), item)
            if item=='stimu_0.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(3) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_1.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(4) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_2.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(7) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_3.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(2) + '.jpg')
                os.rename(src, dst) 
            if item=='stimu_4.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(1) + '.jpg')  
                os.rename(src, dst)
            if item=='stimu_5.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(5) + '.jpg') 
                os.rename(src, dst)  
            if item=='stimu_6.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(6) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_7.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(0) + '.jpg')  
                os.rename(src, dst)

    if sub==5:
        for item in filelist:
            src = os.path.join(os.path.abspath(src_path), item)
            if item=='stimu_0.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(1) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_1.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(2) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_2.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(3) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_3.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(7) + '.jpg')
                os.rename(src, dst) 
            if item=='stimu_4.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(0) + '.jpg')  
                os.rename(src, dst)
            if item=='stimu_5.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(6) + '.jpg')   
                os.rename(src, dst)
            if item=='stimu_6.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(4) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_7.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(5) + '.jpg') 
                os.rename(src, dst) 

    if sub==6:
        for item in filelist:
            src = os.path.join(os.path.abspath(src_path), item)
            if item=='stimu_0.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(2) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_1.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(5) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_2.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(0) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_3.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(7) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_4.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(6) + '.jpg')  
                os.rename(src, dst)
            if item=='stimu_5.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(3) + '.jpg') 
                os.rename(src, dst)  
            if item=='stimu_6.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(1) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_7.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(4) + '.jpg') 
                os.rename(src, dst) 

    if sub==7:
        for item in filelist:
            src = os.path.join(os.path.abspath(src_path), item)
            if item=='stimu_0.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(4) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_1.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(0) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_2.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(6) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_3.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(1) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_4.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(7) + '.jpg')
                os.rename(src, dst)  
            if item=='stimu_5.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(3) + '.jpg')
                os.rename(src, dst)   
            if item=='stimu_6.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(2) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_7.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(5) + '.jpg')  
                os.rename(src, dst)

    if sub==8:
        for item in filelist:
            src = os.path.join(os.path.abspath(src_path), item)
            if item=='stimu_0.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(1) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_1.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(7) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_2.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(0) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_3.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(3) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_4.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(5) + '.jpg') 
                os.rename(src, dst) 
            if item=='stimu_5.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(4) + '.jpg') 
                os.rename(src, dst)  
            if item=='stimu_6.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(2) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_7.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(6) + '.jpg')  
                os.rename(src, dst)

    if sub==9:
        for item in filelist:
            src = os.path.join(os.path.abspath(src_path), item)
            if item=='stimu_0.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(1) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_1.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(2) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_2.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(5) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_3.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(4) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_4.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(0) + '.jpg') 
                os.rename(src, dst) 
            if item=='stimu_5.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(7) + '.jpg') 
                os.rename(src, dst)  
            if item=='stimu_6.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(3) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_7.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(6) + '.jpg') 
                os.rename(src, dst) 

    if sub==10:
        for item in filelist:
            src = os.path.join(os.path.abspath(src_path), item)
            if item=='stimu_0.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(4) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_1.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(5) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_2.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(1) + '.jpg')
                os.rename(src, dst)
            if item=='stimu_3.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(3) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_4.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(7) + '.jpg') 
                os.rename(src, dst) 
            if item=='stimu_5.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(2) + '.jpg') 
                os.rename(src, dst)  
            if item=='stimu_6.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(6) + '.jpg') 
                os.rename(src, dst)
            if item=='stimu_7.jpg':
                dst = os.path.join(os.path.abspath(dst_path), str(0) + '.jpg')
                os.rename(src, dst)

for i in range(1,7):
    for j in range(1,11):
        src_path='data/person_'+str(i)+'/sub_'+str(j)+'/'
        dst_path='data_tran/person_'+str(i)+'/sub_'+str(j)+'/'
        if not os.path.isdir(dst_path):
            os.makedirs(dst_path)
        renaming(src_path,dst_path,j)
  