import click
import yaml
import os

from toolbox.plot import plot
from toolbox.settings import BASE_DIR


@click.command()
@click.argument('experiment', type=str, required=True)
def main(experiment):
    plot_dict = yaml.load(open(os.path.join(BASE_DIR, 'experiments',
                                            'plot.yml')),
                          Loader=yaml.FullLoader)
    exp_dict = yaml.load(open(os.path.join(BASE_DIR, 'experiments',
                                           experiment)),
                         Loader=yaml.FullLoader)
    savedir = plot_dict.pop('savedir')
    for exp_paths, plot_name in zip(exp_dict['paths'], exp_dict['plot_name']):
        print('Processing {} located in {} ...'.format(plot_name, exp_paths))
        for prefix, scalar_key, file_key, kwargs_plot in exp_dict['keys']:
            print('{}/{}'.format(prefix, scalar_key))
            plot_dict['logs_paths'] = exp_paths
            plot_dict['prefix'] = prefix
            plot_dict['key'] = scalar_key
            plot_dict['output'] = os.path.join(savedir, '{}_{}.png'.format(plot_name, file_key))
            key_plot_dict = plot_dict.copy()
            if kwargs_plot is not None:
                key_plot_dict.update(kwargs_plot)
            plot(**key_plot_dict)
        print('Plots saved in {}/{}_*.png'.format(savedir, plot_name))


if __name__ == '__main__':
    main()
