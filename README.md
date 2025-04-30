# Examen DVC and Dagshub
```bash       
├── examen_dvc          
│   ├── data       
│   │   ├── raw      
│   │   ├── processed
│   │   └── predicted
│   ├── metrics       
│   ├── models      
│   ├── src       
│   │   ├── data
│   │   └── models        
│   └── README.md.py       
```

# How to run the project
Run this command to init and run the pipeline:
```bash
dvc init
dvc repro
```

Display the pipeline:
```bash
dvc dag
```