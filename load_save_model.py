import torch
import os
import shutil

def load_model(model,Model_name,Train_mode,Dataset):
    Model_name=Model_name.upper()
    Train_mode=Train_mode.upper()
    Dataset =Dataset.upper()
    filepath= os.getcwd()+'/'+Dataset+'/'+Train_mode+'/'+Model_name+'/'+Model_name+'_'+Train_mode+'_'+Dataset+'.t7'
    #############################
    print('File to be loaded:'+filepath)
    if os.path.isfile(filepath):
            try:
                model=model.module #For DATAPARALLEL
            except:
                pass 
            print('Loading File: '+filepath)
            model.load_state_dict(torch.load(filepath))
            return model
    else:
            print ('WARNING!!!: Weight of '+Model_name+' not loaded. No Existing file')
            return model


def save_model(model,trainAcc_to_file,testAcc_to_file,trainloss_to_file,testloss_to_file,Parameters,
               Model_name,Train_mode,Dataset,model2=None,**kwargs):
        try:
            model=model.module
        except:
            pass

        path= kwargs['save_path']
        if not os.path.exists(path):
            os.makedirs(path)

        stage=''
        if model2 is not None:
            weights_filename1=Model_name+'_'+Train_mode+'_'+Dataset+'_1.t7'
            weights_filename2=Model_name+'_'+Train_mode+'_'+Dataset+'_2.t7'
            torch.save(model.state_dict(),path+weights_filename1)
            torch.save(model2.state_dict(),path+weights_filename2)
        else:
            weights_filename=Model_name+'_'+Train_mode+'_'+Dataset+'.t7'
            torch.save(model.state_dict(),path+weights_filename)
            print(path+weights_filename+' saved')
        
        if testAcc_to_file is not None:
            testacc_filename='Testacc_'+stage+Model_name+'_'+Train_mode+'_'+Dataset+'.csv'
            if os.path.isfile(path+testacc_filename):
                thefile = open(path+testacc_filename, 'a')
            else:
                thefile = open(path+testacc_filename, 'w')
            for item in testAcc_to_file:
                thefile.write("%s," % item)
            thefile.close()
        
        if testloss_to_file is not None:
            testloss_filename='Testloss_'+stage+Model_name+'_'+Train_mode+'_'+Dataset+'.csv'
            if os.path.isfile(path+testloss_filename):
                thefile = open(path+testloss_filename, 'a')
            else:
                thefile = open(path+testloss_filename, 'w')
            for item in testloss_to_file:
                thefile.write("%s," % item)
            thefile.close() 
        
        if trainloss_to_file is not None:
            trainloss_filename='Trainloss_'+stage+Model_name+'_'+Train_mode+'_'+Dataset+'.csv'
            if os.path.isfile(path+trainloss_filename):
                thefile = open(path+trainloss_filename, 'a')
            else:
                thefile = open(path+trainloss_filename, 'w')
            for item in trainloss_to_file:
                thefile.write("%s," % item)
            thefile.close() 
        
        if trainAcc_to_file is not None:
            trainacc_filename='Trainacc_'+stage+Model_name+'_'+Train_mode+'_'+Dataset+'.csv'
            if os.path.isfile(path+trainacc_filename):
                thefile = open(path+trainacc_filename, 'a')
            else:
                thefile = open(path+trainacc_filename, 'w')
            for item in trainAcc_to_file:
                thefile.write("%s," % item)
            thefile.close() 
        
        param_filename='Parameters_'+Model_name+'_'+Train_mode+'_'+Dataset+'.txt'
        if os.path.isfile(path+param_filename):
            thefile = open(path+param_filename, 'a')
        else:
            thefile = open(path+param_filename, 'w')
        thefile.write('%s \n' %stage)
        thefile.write("Patience_scheduler=%s,  Weight_decay=%s  \n" %(Parameters[2],Parameters[3]))
        if not Parameters[1][0][1:] == Parameters[1][0][:-1]:
            for i in range(len(Parameters[1][0])):
                thefile.write("Initial learning rate for param_groups %s is %s epochs \n" %(str(i),Parameters[1][0][i]))
        else:
            thefile.write("Initial learning rate is %s epochs \n" %Parameters[1][0][0])
        thefile.write("\n\n" )

        for epoch,lr in zip(Parameters[0],Parameters[1][1:]):
            thefile.write("In epoch %s, maximum of the learning rates decreased to %s \n" %(epoch, lr))
        thefile.write("Trained for %s epochs \n\n" %Parameters[0][-1])

        thefile.write("Train Statistics \n")
        if trainAcc_to_file is not None:
            thefile.write('Accuracy: %s \n' %trainAcc_to_file[-1])
        thefile.write('Average Loss: %s \n\n'%trainloss_to_file[-1])
        
        thefile.write("Test Statistics \n")
        if testAcc_to_file is not None:
            thefile.write('Accuracy: %s \n' %testAcc_to_file[-1])
            for i in range(len(testAcc_to_file)):
                if testAcc_to_file[i]==testAcc_to_file[-1]:
                    break
            if i+1==len(testAcc_to_file):
                i=-1
            thefile.write('Maximum test accuracy in epoch %s (if 0  it means that the initial state was the best)\n\n'%str(i+1))

        thefile.write('Average Loss: %s \n\n'%testloss_to_file[-1])
        thefile.write('Total time elapsed %s\n\n' %Parameters[4])
        thefile.write('Note: %s\n\n' %kwargs['additional_notes'])
        thefile.write(20*'-'+'\n\n')
        thefile.close() 
        print(os.getcwd()+'/CHECKPOINT/checkpoint_'+Model_name+'_'+Train_mode+'_'+Dataset)
        shutil.rmtree(os.getcwd()+'/CHECKPOINT/checkpoint_'+Model_name+'_'+Train_mode+'_'+Dataset)


'''
def checkpoint_save(model,trainAcc_to_file,testAcc_to_file,trainloss_to_file,testloss_to_file,Parameters,Model_name,Train_mode,Dataset):
        
        path=os.getcwd()+'/CHECKPOINT/checkpoint_'+Model_name+'_'+Train_mode+'_'+Dataset
        if not os.path.exists(path):
            os.makedirs(path)
            
        torch.save(model.state_dict(),path+'/CHECKPOINT.t7')
        print(path+'/CHECKPOINT.t7'+' saved')  

        thefile = open(path+'/Testacc_CHECKPOINT.csv', 'w')
        for item in testAcc_to_file:
            thefile.write("%s," % item)
        thefile.close()
    
        
        thefile = open(path+'/Testloss_CHECKPOINT.csv', 'w')
        for item in testloss_to_file:
            thefile.write("%s," % item)
        thefile.close() 
        
        thefile = open(path+'/Trainloss_CHECKPOINT.csv', 'w')
        for item in trainloss_to_file:
            thefile.write("%s," % item)
        thefile.close() 
        
        
        thefile = open(path+'/Trainacc_CHECKPOINT.csv', 'w')
        for item in trainAcc_to_file:
            thefile.write("%s," % item)
        thefile.close() 
        
        thefile = open(path+'/Parameters_CHECKPOINT.txt', 'w')
            
        thefile.write("Patience_scheduler=%s,  Weight_decay=%s  \n" %(Parameters[2],Parameters[3]))       
        if not Parameters[1][0][1:] == Parameters[1][0][:-1]:
            for i in range(len(Parameters[1][0])):
                thefile.write("Initial learning rate for param_grooups %s is %s epochs \n" %(str(i),Parameters[1][0][i]))
        else:
            thefile.write("Initial learning rate is %s epochs \n" %Parameters[1][0][0])
            
        for epoch,lr in zip(Parameters[0],Parameters[1][1:]):
            thefile.write("In epoch %s, maximum learning rate decreased to %s \n" %(epoch, lr))
        if not(Parameters[0]==[]):
            thefile.write("Trained for %s epochs \n" %Parameters[0][-1])  
        thefile.write("\n\n" )    
        if not(trainAcc_to_file==[]):
            thefile.write("Train Statistics \n")
            thefile.write('Accuracy: %s \n' %trainAcc_to_file[-1])
            thefile.write('Average Loss: %s \n\n'%trainloss_to_file[-1])
        
            thefile.write("Test Statistics \n")
            thefile.write('Accuracy: %s \n' %testAcc_to_file[-1])
            thefile.write('Average Loss: %s \n\n'%testloss_to_file[-1])
            thefile.write(20*'-'+'\n\n')
        thefile.close() 


######################################################################################################
'''        
def checkpoint_save_stage(model,trainloss_to_file,testloss_to_file,train_metric_to_file,test_metric_to_file,Parameters,Model_name,Train_mode,Dataset,model2=None):
        
        path=os.getcwd()+'/CHECKPOINT/checkpoint_'+Model_name+'_'+Train_mode+'_'+Dataset
        if not os.path.exists(path):
            os.makedirs(path)

        if model2 is not None:
            torch.save(model.state_dict(),path+'/CHECKPOINT1.t7')
            torch.save(model2.state_dict(),path+'/CHECKPOINT2.t7')
        else:
            torch.save(model.state_dict(),path+'/CHECKPOINT.t7')
        print(path+'/CHECKPOINT.t7'+' saved')  
    
        thefile = open(path+'/Testloss_CHECKPOINT.csv', 'w')
        for item in testloss_to_file:
            thefile.write("%s," % item)
        thefile.close() 
        
        thefile = open(path+'/Trainloss_CHECKPOINT.csv', 'w')
        for item in trainloss_to_file:
            thefile.write("%s," % item)
        thefile.close() 
        
        thefile = open(path+'/Parameters_CHECKPOINT.txt', 'w')
        thefile.write("STAGE1 \n" )     
        thefile.write("Patience_scheduler=%s,  Weight_decay=%s  \n" %(Parameters[2],Parameters[3]))       
        if not Parameters[1][0][1:] == Parameters[1][0][:-1]:
            for i in range(len(Parameters[1][0])):
                thefile.write("Initial learning rate for param_groups %s is %s epochs \n" %(str(i),Parameters[1][0][i]))
        else:
            thefile.write("Initial learning rate is %s epochs \n" %Parameters[1][0][0])
            
        for epoch,lr in zip(Parameters[0],Parameters[1][1:]):
            thefile.write("In epoch %s, maximum learning rate decreased to %s \n" %(epoch, lr))
        if not(Parameters[0]==[]):
            thefile.write("Trained for %s epochs \n" %Parameters[0][-1])  
        thefile.write("\n\n" )    
        if not(trainloss_to_file==[]):
            thefile.write("Train Statistics \n")
            thefile.write('Accuracy: %s \n' %train_metric_to_file[-1])
            thefile.write('Average Loss: %s \n\n'%trainloss_to_file[-1])
        
            thefile.write("Test Statistics \n")
            thefile.write('Accuracy: %s \n' %test_metric_to_file[-1])
            thefile.write('Average Loss: %s \n\n'%testloss_to_file[-1])
            thefile.write(20*'-'+'\n\n')
        thefile.close() 
