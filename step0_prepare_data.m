% This file is used to prepare the data for the pipeline

YourPattern = '*';
YourOnyen = 'yuukias';
home = '/work/users/y/u/yuukias/Strain/Data' % todo1: modify this
out = sprintf('/work/users/y/u/yuukias/Strain/Data/out'); % todo2: modify this

long_axis = home + "/20208/"
short_axis = home + "/20209/"
aortic = home + "/20210/"
tagging = home + "/20211/"
LVOT = home + "/20212/"
blood_flow = home + "/20213/"
T1 = home + "/20214/"

sub_la = dir(sprintf('%s/20208/%s.zip', home, YourPattern)); sub_la = {sub_la.name}';
sub_la = cellfun(@(x)strsplit(x, '_'), sub_la, 'UniformOutput', 0);
sub_la = cellfun(@(x)x{1}, sub_la, 'UniformOutput', 0);

sub_sa = dir(sprintf('%s/20209/%s.zip', home, YourPattern)); sub_sa = {sub_sa.name}';
sub_sa = cellfun(@(x)strsplit(x, '_'), sub_sa, 'UniformOutput', 0);
sub_sa = cellfun(@(x)x{1}, sub_sa, 'UniformOutput', 0);

sub_aor = dir(sprintf('%s/20210/%s.zip', home, YourPattern)); sub_aor = {sub_aor.name}';
sub_aor = cellfun(@(x)strsplit(x, '_'), sub_aor, 'UniformOutput', 0);
sub_aor = cellfun(@(x)x{1}, sub_aor, 'UniformOutput', 0);

sub_tag = dir(sprintf('%s/20211/%s.zip', home, YourPattern)); sub_tag = {sub_tag.name}';
sub_tag = cellfun(@(x)strsplit(x, '_'), sub_tag, 'UniformOutput', 0);
sub_tag = cellfun(@(x)x{1}, sub_tag, 'UniformOutput', 0);

sub_lvot = dir(sprintf('%s/20212/%s.zip', home, YourPattern)); sub_lvot = {sub_lvot.name}';
sub_lvot = cellfun(@(x)strsplit(x, '_'), sub_lvot, 'UniformOutput', 0);
sub_lvot = cellfun(@(x)x{1}, sub_lvot, 'UniformOutput', 0);

sub_blood = dir(sprintf('%s/20213/%s.zip', home, YourPattern)); sub_blood = {sub_blood.name}';
sub_blood = cellfun(@(x)strsplit(x, '_'), sub_blood, 'UniformOutput', 0);
sub_blood = cellfun(@(x)x{1}, sub_blood, 'UniformOutput', 0);

sub_t1 = dir(sprintf('%s/20214/%s.zip', home, YourPattern)); sub_t1 = {sub_t1.name}';
sub_t1 = cellfun(@(x)strsplit(x, '_'), sub_t1, 'UniformOutput', 0);
sub_t1 = cellfun(@(x)x{1}, sub_t1, 'UniformOutput', 0);

length_la = length(sub_la);
length_sa = length(sub_sa);
length_aor = length(sub_aor);
length_tag = length(sub_tag);
length_lvot = length(sub_lvot);
length_blood = length(sub_blood);
length_t1 = length(sub_t1);

% * Modify here to select the subjects you want to use
sub_used = sub_la;
% sub_used = intersect(sub_la, sub_sa);
% sub_used = intersect(intersect(sub_la, sub_sa), sub_aor);  % todo3: enable if want to use aortic data
nn = length(sub_used);
system(sprintf('mkdir -p %s', out))

Home = '/work/users/y/u/yuukias/Heart_pipeline';  
envdir = '/work/users/y/u/yuukias/env/bin/activate' 
codedir=fullfile(Home,'code');
mkdir(codedir);
delete(sprintf('%s/*',codedir));
fid2 = fopen(sprintf('%s/batAll.sh', codedir), 'w');
N = 5;
K = ceil(nn / N);

% #!/bin/bash
% #SBATCH --ntasks=1
% #SBATCH --job-name=
% #SBATCH --cpus-per-task=4
% #SBATCH --mem=32G
% #SBATCH --time=24:00:00
% #SBATCH --partition=general
% #SBATCH --output=.out

for kk = 1:K
    fid21 = fopen(sprintf('%s/bat%i.pbs', codedir, kk), 'w');
    fprintf(fid21, '#!/bin/bash\n');
    fprintf(fid21, '#SBATCH --ntasks=1\n');
    fprintf(fid21, '#SBATCH --time=23:59:59\n');
    fprintf(fid21, '#SBATCH --mem=12000\n');
    fprintf(fid21, '#SBATCH --wrap=TBSS_%i\n', kk);
    fprintf(fid21, 'module load python/3.9.6\n');
    fprintf(fid21, 'source %s\n', envdir);
    flag = 0;

    for ii = min(kk * N, nn) + ((kk - 1) * N + 1) - (((kk - 1) * N + 1):min(kk * N, nn))
        ID = sub_used{ii};
        fprintf(fid21, 'python /work/users/y/u/yuukias/Heart_pipeline/script/step0_prepare_data.py %s %s %s %s %s\n', out, ID, long_axis, short_axis, aortic)
        flag = 1;
    end;

    fclose(fid21);

    if flag == 1
        fprintf(fid2, 'sbatch bat%i.pbs\n', kk);
    end

    % system(sprintf('python /work/users/y/u/yuukias/Heart_pipeline/script/step0_dataPrepare.py %s %s %s %s %s', out, sub_used{i}, long_axis, short_axis, aortic));
    % system(sprintf('python /work/users/y/u/yuukias/Heart_pipeline/data/step0_dataPrepare_retest.py %s %s',out,sub_used{i}));
end
