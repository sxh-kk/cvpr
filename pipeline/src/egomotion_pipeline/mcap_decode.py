"""Embedded Protobuf decoding for original camera access units and capture time."""

from pathlib import Path


def decoded_messages(path, topics):
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    with Path(path).open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        # File order keeps large compressed payloads out of a log-time sorting heap.
        for schema, channel, message, decoded in reader.iter_decoded_messages(topics=topics, log_time_order=False):
            if schema is None or channel.message_encoding != "protobuf" or schema.encoding != "protobuf":
                raise ValueError(f"{channel.topic}: expected an embedded Protobuf schema")
            yield channel.topic, message, decoded


def timestamp_ns(decoded, message):
    if "header" in decoded.DESCRIPTOR.fields_by_name and decoded.HasField("header"):
        return int(decoded.header.ts), "header.ts"
    if "timestamp" in decoded.DESCRIPTOR.fields_by_name and decoded.HasField("timestamp"):
        stamp = decoded.timestamp
        return int(stamp.seconds) * 1_000_000_000 + int(stamp.nanos), "timestamp"
    return int(message.log_time), "log_time_fallback"
