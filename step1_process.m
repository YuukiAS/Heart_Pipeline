%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%   Process FA data using FSL/TBSS_1 on BIOS Sever     %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% March  8, 2016 @ by CH

clear all;

Home = '/work/users/y/u/yuukias/Heart_pipeline';
cd(Home);
datadir = '/work/users/y/u/yuukias/database/UKBiobank/data_reference/out/nii'; %todo1: modify this
envdir = '/work/users/y/u/yuukias/env/bin/activate'
fpressure = "/work/users/y/u/yuukias/Heart_pipeline/demo_csv/blood_pressure_img.csv"
codedir = fullfile(Home, 'code');
mkdir(codedir);
delete(sprintf('%s/*', codedir));

subNames = dir(datadir);
subNames = {subNames.name}';
subNames = subNames(3:end);

nn = size(subNames, 1);

fid2 = fopen(sprintf('%s/batAll.sh', codedir), 'w');
fprintf(fid2, '#!/bin/bash\n');
N = 5;
K = ceil(nn / N);

for kk = 1:K
    fid21 = fopen(sprintf('%s/bat%i.pbs', codedir, kk), 'w');
    fprintf(fid21, '#!/bin/bash\n');
    fprintf(fid21, '#SBATCH --ntasks=1\n');
    fprintf(fid21, '#SBATCH --job-name=Heart_pipeline_process_%i\n', kk);
    fprintf(fid21, '#SBATCH --time=23:59:59\n');
    fprintf(fid21, '#SBATCH --mem=16G\n');
    fprintf(fid21, '#SBATCH --output=Heart_pipeline_process_%i\n', kk);
    fprintf(fid21, 'module load python/3.9.6\n');
    fprintf(fid21, 'cd %s\n', Home);
    fprintf(fid21, 'PATH=%s:%s/third_party/ubuntu_18.04_bin:${PATH}\n', Home, Home);
    fprintf(fid21, 'module load mirtk/2.0.0;module load gcc/6.3.0\n');
    fprintf(fid21, 'PYTHONPATH=%s\n', Home); % so that we can access 'common' folder
    flag = 0;
    fprintf(fid21, 'source %s\n', envdir);
    for ii = min(kk * N, nn) + ((kk - 1) * N + 1) - (((kk - 1) * N + 1):min(kk * N, nn))
        disp(ii) % display
        ID = subNames{ii};
        sub0temp = dir(sprintf('%s/%s/out/table*.csv', datadir, ID));

        if length(sub0temp) > 5 % all tables already exist
            continue;
        end
        
        fprintf(fid21, 'python3 -u script/step1_process.py %s %s %s\n', datadir, ID, fpressure);
        flag = 1;
    end

    fclose(fid21);

    if flag == 1
        fprintf(fid2, 'sbatch bat%i.pbs\n', kk);
    end

end

fclose(fid2);
