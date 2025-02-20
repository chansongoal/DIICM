set cluster=C31
set hpcshare=\\dellnas2\hpc31\data\gaochangsheng

set project_name=%1
set cfg_name=%2

set width=%3
set height=%4
set QP=%5
set seq_name=%6

set mask_type=%7
set mask_network=%8
set alpha=%9

set hpcbat=\\dellnas2\hpc31\users\gaochangsheng\%project_name%
set hpcout=%hpcshare%\out\%project_name%
set hpcerr=%hpcshare%\log_unused\%project_name%\


@rem set nodes=C01,C02,C03,C04,C05,C07,C08,C09,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25, /askednodes:%nodes% 
@rem set nodes=C32,C33,C34,C35,C36,C37,C38,C39,C40


for /F "tokens=4" %%j in ('job new /jobname:%seq_name%_%mask_type%_%alpha%_QP%QP% /scheduler:%cluster%') do set JOBID=%%j
job add %JOBID% /scheduler:%cluster% /stderr:%hpcerr%\%seq_name%_QP%QP%.txt /name:%seq_name%_QP%QP%  /workdir:%hpcout% %hpcbat%\vtm_anchor_transformed.bat %project_name% %cfg_name% %width% %height% %QP% %seq_name% %mask_type% %mask_network% %alpha%
job submit /scheduler:%cluster% /id:%JOBID% /priority:4000 /numcores:1 /user:bit\dongcunhui  /password:dongcunhui.916