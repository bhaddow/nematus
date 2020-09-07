import logging
import sys
import time
import types
import numpy
import tensorflow as tf
numpy.random.seed(111)
tf.set_random_seed(111)
# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from . import exception
    from . import util
except (ModuleNotFoundError, ImportError) as e:
    import exception
    import util

# nparr = []
def translate_batch_slt(session, sampler, x, x_mask, max_translation_len,
                    normalization_alpha, last_translation = None):
    """Translate a batch using a RandomSampler or BeamSearchSampler.

    Args:
        session: a TensorFlow session.
        sampler: a BeamSearchSampler or RandomSampler object.
        x: input Tensor with shape (factors, max_seq_len, batch_size).
        x_mask: mask Tensor for x with shape (max_seq_len, batch_size).
        max_translation_len: integer specifying maximum translation length.
        normalization_alpha: float specifying alpha parameter for length
            normalization.

    Returns:
        A list of lists of (translation, score) pairs. The outer list contains
        one list for each input sentence in the batch. The inner lists contain
        k elements (where k is the beam size), sorted by score in best-first
        order.
    """
    # print('last_translation:\t',last_translation)

    x_tiled = numpy.tile(x, reps=[1, 1, sampler.beam_size])
    x_mask_tiled = numpy.tile(x_mask, reps=[1, sampler.beam_size])

    feed_dict = {}

    # Feed inputs to the models.
    for model, config in zip(sampler.models, sampler.configs):
        if config.model_type == 'rnn':
            feed_dict[model.inputs.x] = x_tiled
            feed_dict[model.inputs.x_mask] = x_mask_tiled
        else:
            assert config.model_type == 'transformer'
            # Inputs don't need to be tiled in the Transformer because it
            # checks for different batch sizes in the encoder and decoder and
            # does its own tiling internally at the connection points.
            feed_dict[model.inputs.x] = x
            feed_dict[model.inputs.x_mask] = x_mask
        feed_dict[model.inputs.training] = False

    # Feed inputs to the sampler.
    feed_dict[sampler.inputs.batch_size_x] = x.shape[-1]
    feed_dict[sampler.inputs.max_translation_len] = max_translation_len
    feed_dict[sampler.inputs.normalization_alpha] = normalization_alpha
    feed_dict[sampler.inputs.last_translation] = last_translation
    feed_dict[sampler.inputs.last_translation_len] = len(last_translation)
    # Run the sampler.
    translations, scores = session.run(sampler.outputs, feed_dict=feed_dict)
    # print(type(translations))
    assert len(translations) == x.shape[-1]
    assert len(scores) == x.shape[-1]
    # nparr.append(scores)
    # print(scores)
    # Sort the translations by score. The scores are (optionally normalized)
    # log probs so higher values are better.
    beams = []
    for i in range(len(translations)):
        pairs = zip(translations[i], scores[i])
        beams.append(sorted(pairs, key=lambda pair: pair[1], reverse=True))

    return beams

def translate_batch(session, sampler, x, x_mask, max_translation_len,
                    normalization_alpha):
    """Translate a batch using a RandomSampler or BeamSearchSampler.

    Args:
        session: a TensorFlow session.
        sampler: a BeamSearchSampler or RandomSampler object.
        x: input Tensor with shape (factors, max_seq_len, batch_size).
        x_mask: mask Tensor for x with shape (max_seq_len, batch_size).
        max_translation_len: integer specifying maximum translation length.
        normalization_alpha: float specifying alpha parameter for length
            normalization.

    Returns:
        A list of lists of (translation, score) pairs. The outer list contains
        one list for each input sentence in the batch. The inner lists contain
        k elements (where k is the beam size), sorted by score in best-first
        order.
    """

    x_tiled = numpy.tile(x, reps=[1, 1, sampler.beam_size])
    x_mask_tiled = numpy.tile(x_mask, reps=[1, sampler.beam_size])

    feed_dict = {}

    # Feed inputs to the models.
    for model, config in zip(sampler.models, sampler.configs):
        if config.model_type == 'rnn':
            feed_dict[model.inputs.x] = x_tiled
            feed_dict[model.inputs.x_mask] = x_mask_tiled
        else:
            assert config.model_type == 'transformer'
            # Inputs don't need to be tiled in the Transformer because it
            # checks for different batch sizes in the encoder and decoder and
            # does its own tiling internally at the connection points.
            feed_dict[model.inputs.x] = x
            feed_dict[model.inputs.x_mask] = x_mask
        feed_dict[model.inputs.training] = False

    # Feed inputs to the sampler.
    feed_dict[sampler.inputs.batch_size_x] = x.shape[-1]
    feed_dict[sampler.inputs.max_translation_len] = max_translation_len
    feed_dict[sampler.inputs.normalization_alpha] = normalization_alpha

    # Run the sampler.
    translations, scores = session.run(sampler.outputs, feed_dict=feed_dict)

    assert len(translations) == x.shape[-1]
    assert len(scores) == x.shape[-1]

    # Sort the translations by score. The scores are (optionally normalized)
    # log probs so higher values are better.
    beams = []
    for i in range(len(translations)):
        pairs = zip(translations[i], scores[i])
        beams.append(sorted(pairs, key=lambda pair: pair[1], reverse=True))

    return beams


def translate_file(input_file, output_file, session, sampler, config,
                   max_translation_len, normalization_alpha, nbest=False,
                   minibatch_size=80, maxibatch_size=20, strategy='biased_beam_search'):
    """Translates a source file using a RandomSampler or BeamSearchSampler.

    Args:
        input_file: file object from which source sentences will be read.
        output_file: file object to which translations will be written.
        session: TensorFlow session.
        sampler: BeamSearchSampler or RandomSampler object.
        config: model config.
        max_translation_len: integer specifying maximum translation length.
        normalization_alpha: float specifying alpha parameter for length
            normalization.
        nbest: if True, produce n-best output with scores; otherwise 1-best.
        minibatch_size: minibatch size in sentences.
        maxibatch_size: number of minibatches to read and sort, pre-translation.
    """

    def translate_maxibatch(maxibatch, num_to_target, num_prev_translated, mask=0):
        """Translates an individual maxibatch.

        Args:
            maxibatch: a list of sentences.
            num_to_target: dictionary mapping target vocabulary IDs to strings.
            num_prev_translated: the number of previously translated sentences.
        """

        # Sort the maxibatch by length and split into minibatches.
        try:
            minibatches, idxs = util.read_all_lines(config, maxibatch,
                                                    minibatch_size)
        except exception.Error as x:
            logging.error(x.msg)
            sys.exit(1)

        # Translate the minibatches and store the resulting beam (i.e.
        # translations and scores) for each sentence.
        beams = []
        for x in minibatches:
            y_dummy = numpy.zeros(shape=(len(x),1))
            x, x_mask, _, _ = util.prepare_data(x, y_dummy, config.factors,
                                                maxlen=None)
            sample = translate_batch(session, sampler, x, x_mask,
                                     max_translation_len, normalization_alpha)
            beams.extend(sample)
            num_translated = num_prev_translated + len(beams)
            logging.info('Translated {} sents'.format(num_translated))

        # Put beams into the same order as the input maxibatch.
        tmp = numpy.array(beams, dtype=numpy.object)
        ordered_beams = tmp[idxs.argsort()]

        # Write the translations to the output file.
        for i, beam in enumerate(ordered_beams):
            if nbest:
                num = num_prev_translated + i
                for sent, cost in beam:
                    translation = util.seq2words(sent, num_to_target)
                    line = "{} ||| {} ||| {}\n".format(num, translation,
                                                       str(cost))
                    output_file.write(line)
            else:
                best_hypo, cost = beam[0]
                # print(best_hypo)
                eos_idx = list(best_hypo).index(0) if 0 in best_hypo else len(best_hypo)
                best_hypo = best_hypo[:eos_idx]
                best_hypo = best_hypo[:len(best_hypo)-mask] if len(best_hypo) > mask else [] 
                best_hypo = list(best_hypo)+[0]
                # print(best_hypo)
                line = util.seq2words(best_hypo, num_to_target) + '\n'
                output_file.write(line)


    def translate_maxibatch_slt(maxibatch, num_to_target, num_prev_translated, last_translation, mask=0):
        """Translates an individual maxibatch.

        Args:
            maxibatch: a list of sentences.
            num_to_target: dictionary mapping target vocabulary IDs to strings.
            num_prev_translated: the number of previously translated sentences.
        """

        # Sort the maxibatch by length and split into minibatches.
        try:
            minibatches, idxs = util.read_all_lines(config, maxibatch,
                                                    minibatch_size)
        except exception.Error as x:
            logging.error(x.msg)
            sys.exit(1)

        # Translate the minibatches and store the resulting beam (i.e.
        # translations and scores) for each sentence.
        beams = []
        for x in minibatches:
            y_dummy = numpy.zeros(shape=(len(x),1))
            x, x_mask, _, _ = util.prepare_data(x, y_dummy, config.factors,
                                                maxlen=None)
            sample = translate_batch_slt(session, sampler, x, x_mask,
                                     max_translation_len, normalization_alpha, last_translation = last_translation)
            # print(type(sample))

            beams.extend(sample)
            num_translated = num_prev_translated + len(beams)
            logging.info('Translated {} sents'.format(num_translated))

        # Put beams into the same order as the input maxibatch.
        tmp = numpy.array(beams, dtype=numpy.object)
        ordered_beams = tmp[idxs.argsort()]
        # Write the translations to the output file.
        for i, beam in enumerate(ordered_beams):
            if nbest:
                num = num_prev_translated + i
                for sent, cost in beam:
                    translation = util.seq2words(sent, num_to_target)
                    line = "{} ||| {} ||| {}\n".format(num, translation,
                                                       str(cost))
                    output_file.write(line)
            else:
                best_hypo, cost = beam[0]
                # print(best_hypo)
                eos_idx = list(best_hypo).index(0) if 0 in best_hypo else len(best_hypo)
                best_hypo = best_hypo[:eos_idx]
                best_hypo = best_hypo[:len(best_hypo)-mask] if len(best_hypo) > mask else [] 
                # print(best_hypo)
                best_hypo = list(best_hypo)+[0]
                line = util.seq2words(best_hypo, num_to_target) + '\n'
                output_file.write(line)
                return list([1])+list(best_hypo[:-1])

    prefix_lines = input_file.readlines()
    input_file.seek(0, 0)
    prefix_mask = []
    last_prefix = ''
    for prefix_line in prefix_lines:
        # print(last_prefix, prefix_line)
        if not last_prefix:
            last_prefix = prefix_line
            continue
        if last_prefix[:-1] in prefix_line:
            prefix_mask.append(0)
        else:
            prefix_mask.append(1)
        last_prefix = prefix_line
    prefix_mask.append(1)

    _, _, _, num_to_target = util.load_dictionaries(config)

    logging.info("NOTE: Length of translations is capped to {}".format(
        max_translation_len))

    start_time = time.time()

    num_translated = 0
    maxibatch = []

    # a very naive implementation for biased beam search which do not allow minibatch translation 
    if strategy == 'biased_beam_search':
        assert minibatch_size == 1
        assert maxibatch_size == 1
        last_translation = [1]
        last_line = ""
        sent_id = 0
        while True:
            line = input_file.readline()
            if line == "":
                if len(maxibatch) > 0:
                    if not last_line or len(last_line) > len(line) or last_line[:] != line[:len(last_line)]:
                        last_translation = [1]
                    translate_maxibatch_slt(maxibatch, num_to_target, num_translated, last_translation)
                    num_translated += len(maxibatch)
                break
            maxibatch.append(line)
            if len(maxibatch) == (maxibatch_size * minibatch_size):
                if not last_line or len(last_line) > len(line) or last_line[:-1] != line[:len(last_line)-1]:
                    # print(len(last_line) > len(line))
                    # print(last_line[:] != line[:len(last_line)])
                    # print(last_line)
                    # print(line)
                    last_translation = [1]
                # if prefix_mask[sent_id]:
                #     print(sent_id)
                #     print('line:'+line)
                #     print('prefix:'+prefix_lines[sent_id])
                if not prefix_mask[sent_id]:
                    # print('mask works!')
                    last_translation = translate_maxibatch_slt(maxibatch, num_to_target, num_translated, last_translation, mask=15)
                else:
                    last_translation = translate_maxibatch_slt(maxibatch, num_to_target, num_translated, last_translation, mask=0)
                # print(last_translation)
                num_translated += len(maxibatch)
                maxibatch = []
            last_line = line
            sent_id += 1
    else:
        sent_id = 0
        while True:
            line = input_file.readline()
            if line == "":
                if len(maxibatch) > 0:
                    translate_maxibatch(maxibatch, num_to_target, num_translated)
                    num_translated += len(maxibatch)
                break
            maxibatch.append(line)
            if len(maxibatch) == (maxibatch_size * minibatch_size):
                if not prefix_mask[sent_id]:
                    translate_maxibatch(maxibatch, num_to_target, num_translated, mask=15)
                else:
                    translate_maxibatch(maxibatch, num_to_target, num_translated, mask=0)
                # translate_maxibatch(maxibatch, num_to_target, num_translated)
                num_translated += len(maxibatch)
                maxibatch = []
            sent_id += 1

    duration = time.time() - start_time
    logging.info('Translated {} sents in {} sec. Speed {} sents/sec'.format(
        num_translated, duration, num_translated/duration))

    output_file.flush()
    # numpy.save('beam_score', nparr)
