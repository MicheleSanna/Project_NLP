#Questo file contiene le variabili che differiscono tra la configurazione da tenere sul
#computer portatile alla configurazione per il cluster dell'università.
#Nel cluster dell'università vi è un'altra copia di questo file, ma con valori differenti
#per ogni variabile (la logica è che un file corto sia più facile da editare con nano)

NUM_WORKERS= 10 #Number of workers for the dataloaders
N_FILE= 1000 #Number of file regarding the dataset subdivision
BATCH_SIZE= 16 #Batch size per process
MASTER_ADDR= "localhost"
MASTER_PORT= "12355"
BACKEND= "nccl"
MULTIPROCESS= True #If multiprocess set True
CHECKPOINT = False #Set true if you have a checkpoint.save to load
N_STEP = 500 #Number of step between every tensorboard and log writing
TRAIN_DATASET = 5000000 #Dimension of training dataset
EVALUATE_DATASET = 4999018 #Dimension of evaluation dataset
EVALUATE_STEPS = 2048 #Number of datapoints to evaluate the dataset
NUM_EPOCHS = 5 #Number of epochs
